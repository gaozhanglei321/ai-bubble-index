"""AI bubble dashboard V3.2 / 完整模型与早期参考历史分开统计。"""
import json
import hashlib
import os
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import percentileofscore
import requests

# 用户配置 / User configuration: complete-model formula is unchanged.
USER_CONFIG = {
    "START_DATE": "1960-01-01",
    "BIAS_POINTS": 15.0,  # 人工风险修正 / Required manual risk adjustment
    "CACHE_SECONDS": 3600,
    "MISSING_POLICY": "legacy",  # legacy 保持原版前向填充；strict 不补值（会改变部分读数）
    "SAVE_LOCAL_BACKUP": True,
    "BACKUP_DIR": "bubble_data",  # Relative to this script, not the working directory
}
MODEL_VERSION = "3.2-active-fund-screener"
TICKERS = ["QQQ", "^VIX", "SPHB", "SPLV", "IPO", "SPY", "HYG", "IEF", "^TNX"]
FACTOR_NAMES = {
    "P1": "QQQ 均线偏离", "P2": "VIX 倒数", "P3": "高低波动比",
    "P4": "IPO 表现与成交量", "P5": "HYG/IEF", "P6": "美债收益率动量",
}
SENTIMENT_WEIGHTS = pd.Series({"P1": .3, "P2": .3, "P3": .1, "P4": .3})
CAPITAL_WEIGHTS = pd.Series({"P5": .5, "P6": .5})
BACKUP_DIR = Path(__file__).resolve().parent / USER_CONFIG["BACKUP_DIR"]


def normalize_raw(raw):
    """Validate a complete download schema without filling missing observations.
    校验字段；上市前和下载缺失值保持为空，不以其他标的代替。
    """
    if raw is None or raw.empty or not isinstance(raw.columns, pd.MultiIndex):
        raise ValueError("未获得有效的多标的行情表。")
    out = raw.copy()
    if "Close" not in out.columns.get_level_values(0):
        if "Close" in out.columns.get_level_values(1):
            out = out.swaplevel(axis=1)
        else:
            raise ValueError("行情缺少 Close 字段。")
    required = [("Close", t) for t in TICKERS] + [("Volume", "IPO")]
    missing = [f"{f}/{t}" for f, t in required
               if (f, t) not in out or out[(f, t)].notna().sum() == 0]
    if missing:
        raise ValueError("行情下载不完整，拒绝按早期模型降级：" + ", ".join(missing))
    out.index = pd.to_datetime(out.index)
    if out.index.tz is not None:
        out.index = out.index.tz_localize(None)
    out.index = out.index.normalize().as_unit("ns")
    if out.index.has_duplicates:
        raise ValueError("行情含重复日期。")
    out = out.sort_index().apply(pd.to_numeric, errors="raise")
    if np.isinf(out.to_numpy(dtype=float)).any():
        raise ValueError("行情包含无穷数值。")
    out.index.name = "Date"
    return out


def backup_bytes(raw):
    """Portable, non-executable archive / 可移植的 CSV+JSON 备份。"""
    raw = normalize_raw(raw)
    csv = raw.to_csv(date_format="%Y-%m-%d").encode("utf-8")
    meta = {
        "schema": 1, "model": MODEL_VERSION, "auto_adjust": True,
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": "Yahoo Finance / yfinance", "config": USER_CONFIG,
        "sha256": hashlib.sha256(csv).hexdigest(),
    }
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("raw.csv", csv)
        z.writestr("metadata.json", json.dumps(meta, ensure_ascii=False, indent=2))
    return buf.getvalue()


def read_backup(content):
    # Read in memory; never extract paths or deserialize executable pickle.
    with zipfile.ZipFile(BytesIO(content)) as z:
        if set(z.namelist()) != {"raw.csv", "metadata.json"}:
            raise ValueError("备份必须包含 raw.csv 和 metadata.json。")
        if any(i.file_size > 80_000_000 for i in z.infolist()):
            raise ValueError("备份文件过大。")
        meta = json.loads(z.read("metadata.json"))
        csv = z.read("raw.csv")
    if meta.get("schema") != 1 or meta.get("auto_adjust") is not True:
        raise ValueError("备份版本或复权口径不匹配。")
    if hashlib.sha256(csv).hexdigest() != meta.get("sha256"):
        raise ValueError("备份校验失败。")
    raw = pd.read_csv(BytesIO(csv), header=[0, 1], index_col=0, parse_dates=True)
    return normalize_raw(raw), meta


def save_snapshot(raw):
    """Immutable snapshots: previous successful downloads are never overwritten.
    独立保存每份变更后的行情快照，避免分红复权的新旧价格拼接。
    """
    payload = backup_bytes(raw)
    digest = hashlib.sha256(raw.to_csv().encode()).hexdigest()[:16]
    BACKUP_DIR.mkdir(parents=True, exist_ok=True)
    name = f"market_{raw.index[-1]:%Y%m%d}_{digest}.zip"
    target = BACKUP_DIR / name
    if not target.exists():
        # Exclusive creation prevents accidental overwrite across app sessions.
        import tempfile
        with tempfile.NamedTemporaryFile(dir=BACKUP_DIR, suffix=".tmp", delete=False) as f:
            temp = Path(f.name)
            f.write(payload)
        os.replace(temp, target)
    return target


def newest_backup():
    if not BACKUP_DIR.exists():
        return None
    for path in sorted(BACKUP_DIR.glob("market_*.zip"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            return read_backup(path.read_bytes())[0], path.name
        except (ValueError, OSError, KeyError, zipfile.BadZipFile):
            continue
    return None


@st.cache_data(ttl=USER_CONFIG["CACHE_SECONDS"], show_spinner=False)
def fetch_market_data():
    # Refresh a consistent adjusted-price snapshot, not an unsafe incremental splice.
    raw = yf.download(TICKERS, start=USER_CONFIG["START_DATE"], interval="1d",
                      auto_adjust=True, actions=True, progress=False, threads=True)
    raw = normalize_raw(raw)
    # The current incomplete US session is not an end-of-day observation.
    us_today = pd.Timestamp.now(tz="America/New_York").date()
    raw = raw.loc[raw.index.date < us_today]
    raw = normalize_raw(raw)
    # A truncated response must not silently replace a longer saved history.
    saved = newest_backup()
    if saved is not None:
        old, _ = saved
        for field, ticker in [("Close", t) for t in TICKERS] + [("Volume", "IPO")]:
            old_valid = old[(field, ticker)].dropna()
            new_valid = raw[(field, ticker)].dropna()
            lost = old_valid.index.difference(new_valid.index)
            if len(lost):
                raise ValueError(f"本次响应缺少已保存的 {ticker}/{field} 历史记录（{len(lost)} 条）；"
                                 "保留旧快照并回退，不拼接不同日期的复权价格。")
    note = ""
    if USER_CONFIG["SAVE_LOCAL_BACKUP"]:
        try:
            save_snapshot(raw)
        except OSError as exc:
            note = f"行情已取得，但服务器本地备份写入失败：{exc}。请下载备份。"
    return raw, note


def rolling_pct(series, window):
    # Equivalent to the original rolling apply/rank, with full-window warmup.
    return series.rolling(window, min_periods=window).rank(method="average", pct=True) * 100


def available_weighted(factors, weights):
    """Normalize only within each module; require at least one valid factor.
    仅参考模型使用可用指标重分配；情绪和资金两个模块缺一不可。
    """
    sub = factors[weights.index]
    denominator = sub.notna().mul(weights, axis=1).sum(axis=1)
    numerator = sub.mul(weights, axis=1).sum(axis=1, min_count=1)
    return numerator / denominator.where(denominator > 0)


def calculate_indices(raw):
    raw = normalize_raw(raw)
    real_qqq = raw[("Close", "QQQ")].notna() & (raw[("Close", "QQQ")] > 0)
    policy = USER_CONFIG["MISSING_POLICY"]
    if policy not in {"legacy", "strict"}:
        raise ValueError("MISSING_POLICY 必须是 legacy 或 strict。")
    # Preserve the original alignment/fill semantics by default. Strict cleanup is
    # an explicit model-data policy change, never a silent part of extending history.
    calendar = raw.index if policy == "legacy" else raw.index[real_qqq]
    observed = raw["Close"].reindex(index=calendar, columns=TICKERS).where(lambda x: x > 0)
    observed_volume = raw[("Volume", "IPO")].reindex(calendar).where(lambda x: x >= 0)
    close = observed.ffill() if policy == "legacy" else observed
    volume = observed_volume.ffill() if policy == "legacy" else observed_volume
    # Forward fill cannot create values before a ticker's first observation.
    factors = pd.DataFrame(index=calendar)
    sma200 = close["QQQ"].rolling(200, min_periods=200).mean()
    factors["P1"] = rolling_pct((close["QQQ"] - sma200) / sma200, 2520)
    factors["P2"] = rolling_pct(1 / close["^VIX"], 2520)
    factors["P3"] = 50 + (rolling_pct(close["SPHB"] / close["SPLV"], 756) - 50) * .4
    vmean = volume.rolling(126, min_periods=126).mean().replace(0, np.nan)
    enhanced = (close["IPO"] / close["SPY"]) * (volume / vmean)
    factors["P4"] = rolling_pct(enhanced, 756)
    p5 = rolling_pct(close["HYG"] / close["IEF"], 756).rolling(10).mean()
    factors["P5"] = (80 - (100 - p5) * 3.0).clip(0, 100)
    change = (close["^TNX"] - close["^TNX"].shift(20)).rolling(10).mean()
    factors["P6"] = pd.Series(np.select(
        [change < -.25, change < -.05, change < .15], [100., 75., 50.], default=25.),
        index=calendar).where(change.notna())
    if policy == "legacy":
        factors["P6"] = factors["P6"].ffill()
    factors = factors.replace([np.inf, -np.inf], np.nan)

    # Full model / 完整模型：原权重、压缩、两次平滑与 +15 均保留。
    s_raw = (factors.P1 * .3 + factors.P2 * .3 + factors.P3 * .1 + factors.P4 * .3)
    s_full = 20 + (s_raw.rolling(10).mean() - 20) * .83
    c_full = (factors.P5 + factors.P6) / 2
    full = (((s_full * 2 + c_full) / 3).rolling(10).mean()
            + USER_CONFIG["BIAS_POINTS"]).clip(0, 100)

    # Early reference / 早期参考：不缩短窗口、不补造历史。
    s_ref = 20 + (available_weighted(factors, SENTIMENT_WEIGHTS).rolling(10).mean() - 20) * .83
    c_ref = available_weighted(factors, CAPITAL_WEIGHTS)
    reference = (((s_ref * 2 + c_ref) / 3).rolling(10).mean()
                 + USER_CONFIG["BIAS_POINTS"]).clip(0, 100)
    first_full = full.first_valid_index()
    # Never reclassify later outages as early history.
    if first_full is not None:
        reference = reference.where(reference.index < first_full)
    display = full.combine_first(reference)
    df = factors.copy()
    df["完整指数"] = full
    df["早期参考指数"] = reference
    df["总泡沫指数"] = display
    df["数据类型"] = np.select([full.notna(), reference.notna()], ["完整模型", "早期参考"], default="不可计算")
    df["综合情绪指标"] = s_full.where(full.notna(), s_ref.where(reference.notna()))
    df["综合资金指标"] = c_full.where(full.notna(), c_ref.where(reference.notna()))
    df["QQQ"] = close["QQQ"]
    df["QQQ_1w_ret"] = close["QQQ"].pct_change(5, fill_method=None)
    df["有效指标数"] = factors.notna().sum(axis=1)
    weights = pd.Series({"P1": .2, "P2": .2, "P3": 1/15, "P4": .2, "P5": 1/6, "P6": 1/6})
    df["原权重覆盖率"] = factors.notna().mul(weights).sum(axis=1) * 100
    df["平滑期最低覆盖率"] = df["原权重覆盖率"].rolling(19).min()
    df["参与指标"] = factors.notna().apply(lambda row: ", ".join(row.index[row]), axis=1)
    filled = (observed.isna() & close.notna()).any(axis=1) | (observed_volume.isna() & volume.notna())
    df["当日沿用旧行情"] = filled
    df["近19日含补值"] = filled.astype(int).rolling(19, min_periods=1).max().astype(bool)
    df.index.name = "日期"
    # Display and return horizons use actual QQQ dates, while original factor
    # calculation calendar is retained in legacy mode for numerical compatibility.
    return df.loc[df.index.intersection(raw.index[real_qqq])]


def percentile_summary(df):
    """Always use full history through the same latest complete date; never view filters."""
    full = df["完整指数"].dropna()
    if full.empty:
        return None
    asof = full.index[-1]
    global_values = df.loc[:asof, "总泡沫指数"].dropna()
    val = float(full.iloc[-1])
    return {"asof": asof, "value": val, "full": full, "global": global_values,
            "full_pct": float(percentileofscore(full, val, kind="rank")),
            "global_pct": float(percentileofscore(global_values, val, kind="rank"))}


@st.cache_data(ttl=3600, show_spinner=False)
def run_backtest(df):
    """Complete-model signal dates only; future returns use the unfiltered QQQ calendar.
    收益周期按真实交易日偏移，不能先删去缺失信号日再计算持有周期。
    """
    results = []
    for lo, hi, label, color in ZONES:
        mask = df["完整指数"].ge(lo) & df["完整指数"].lt(hi)
        row = {"区间": label, "信号天数": int(mask.sum()), "color": color}
        for name, days in PERIODS:
            rets = ((df.QQQ.shift(-days) / df.QQQ - 1) * 100).loc[mask].dropna()
            row.update({f"{name}_avg": rets.mean(), f"{name}_median": rets.median(),
                        f"{name}_win": (rets > 0).mean() * 100 if len(rets) else np.nan,
                        f"{name}_n": len(rets)})
        results.append(row)
    return results


def index_color(value, ret=0):
    if pd.isna(value): return C_MUTED
    # Same [lo, hi) boundary rule as the backtest.
    for lo, hi, _, color in ZONES:
        if lo <= value < hi:
            return "#FF9F0A" if lo == 45 and ret >= 0 else color
    return C_MUTED


def period_text(values):
    values = values.dropna()
    return "暂无有效数据" if values.empty else f"{values.index[0]:%Y-%m-%d} — {values.index[-1]:%Y-%m-%d} · {len(values):,} 个交易日"


def plot_index(frame):
    fig = make_subplots(specs=[[{"secondary_y": True}]])
    for lo, hi, _, color in ZONES:
        fig.add_hrect(y0=lo, y1=min(hi, 100), fillcolor=color, opacity=.045,
                      line_width=0, layer="below", secondary_y=False)
    fig.add_trace(go.Scatter(x=frame.index, y=frame.QQQ, name="QQQ",
        line=dict(color="rgba(209,212,220,.25)", width=1.2),
        hovertemplate="QQQ: $%{y:.2f}<extra></extra>"), secondary_y=True)
    # Mask traces without dropping rows so unavailable periods are never connected.
    colors = pd.Series([index_color(v,r) for v,r in zip(frame["完整指数"], frame.QQQ_1w_ret)], index=frame.index)
    # Connected segments share their boundary point; null values explicitly break lines.
    for color in colors.unique():
        mask = colors.eq(color) & frame["完整指数"].notna()
        y = frame["完整指数"].where(mask | mask.shift(1, fill_value=False))
        fig.add_trace(go.Scatter(x=frame.index, y=y, mode="lines", connectgaps=False,
            line=dict(color=color, width=2.6), hoverinfo="skip", showlegend=False), secondary_y=False)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["完整指数"], mode="lines", name="完整模型",
        line=dict(color="rgba(0,0,0,0)", width=.1), connectgaps=False,
        hovertemplate="完整指数: %{y:.2f}<extra></extra>"), secondary_y=False)
    fig.add_trace(go.Scatter(x=frame.index, y=frame["早期参考指数"], mode="lines", name="早期参考（非完整模型）",
        line=dict(color="#a4a8b5", width=2, dash="dash"), connectgaps=False,
        customdata=frame[["有效指标数", "原权重覆盖率", "参与指标"]].to_numpy(),
        hovertemplate="早期参考: %{y:.2f}<br>当日有效: %{customdata[0]}/6"
                      "<br>原权重覆盖: %{customdata[1]:.1f}%<br>%{customdata[2]}<extra></extra>"), secondary_y=False)
    layout = dark_layout(height=530)
    layout.update(showlegend=True, legend=dict(orientation="h", y=1.12, x=0))
    fig.update_layout(**layout)
    fig.update_yaxes(range=[0,100], title_text="泡沫指数 / 早期参考", secondary_y=False)
    fig.update_yaxes(title_text="QQQ 复权价格", showgrid=False, secondary_y=True)
    return fig


def render_main():
    st.markdown("# 🛡️ 私人量化终端：AI 泡沫综合指数 V3.2")
    st.sidebar.header("⚙️ 看板控制台")
    upload = st.sidebar.file_uploader("从历史备份读取（ZIP）", type=["zip"])
    if st.sidebar.button("重新获取行情"):
        fetch_market_data.clear()
    try:
        if upload is not None:
            raw, meta = read_backup(upload.getvalue())
            st.info(f"当前使用上传备份，创建时间：{meta.get('created_utc', '未知')}。移除文件后恢复在线行情。")
        else:
            try:
                with st.spinner("正在获取完整历史行情…"):
                    raw, note = fetch_market_data()
                if note: st.warning(note)
            except Exception as exc:
                saved = newest_backup()
                if saved is None:
                    raise RuntimeError(f"行情获取失败，且没有可用本地备份：{exc}") from exc
                raw, name = saved
                st.warning(f"在线获取失败，正在使用本地备份 {name}。原因：{exc}")
        df = calculate_indices(raw)
    except Exception as exc:
        st.error(str(exc))
        st.info("可在侧边栏上传以前下载的 ZIP 行情备份；下方基金筛选仍可单独使用。")
        render_fund_screener()
        st.stop()

    stats = percentile_summary(df)
    valid = df[df["总泡沫指数"].notna()]
    if valid.empty:
        st.error("历史长度尚不足以计算任何指数。")
        st.stop()
    full_df = df[df["完整指数"].notna()]
    ref_df = df[df["早期参考指数"].notna()]
    show_early = st.sidebar.checkbox("显示早期参考历史", value=False)
    frame = render_time_controls(df, show_early)
    st.sidebar.caption("时间范围只控制图表；两种百分位始终使用各自全部历史。")

    if stats:
        asof, val = stats["asof"], stats["value"]
        full = stats["full"]
        delta = float(full.iloc[-1] - full.iloc[-2]) if len(full) > 1 else None
        c1,c2,c3 = st.columns([1,2,1])
        c1.metric("完整模型 · AI 泡沫指数", f"{val:.1f}", f"{delta:+.2f}" if delta is not None else None, delta_color="inverse")
        label = next(label for lo,hi,label,_ in ZONES if lo <= val < hi)
        c2.metric("市场状态评级", label)
        c3.metric("完整指数截止日期", f"{asof:%Y-%m-%d}")
        left,right = st.columns(2)
        left.metric("完整期百分位 · 主要", f"{stats['full_pct']:.1f}%")
        left.caption(period_text(stats["full"]))
        right.metric("全局参考百分位 · 辅助", f"{stats['global_pct']:.1f}%")
        right.caption(period_text(stats["global"]) + "；含早期不完整模型")
        if asof < df.index[-1]:
            st.warning(f"最新行情至 {df.index[-1]:%Y-%m-%d}，但完整指数仅有效至 {asof:%Y-%m-%d}。"
                       "后续缺失不转为早期参考；两个百分位统一截止于该有效日期。")
    else:
        st.warning("尚无完整指数；仅能探索参考曲线，不显示完整期或全局百分位，也不开展正式回测。")
    st.caption("完整期要求六项指标及其平滑窗口齐全。早期参考仅在模块内部重分配可用指标权重，"
               "仍保留情绪∶资金=2∶1、原始窗口与 +15 修正。全局百分位混合了不同指标组合，仅供辅助。")
    if USER_CONFIG["MISSING_POLICY"] == "legacy":
        st.caption("当前保持旧版行情处理：上市后的缺口沿用前值，上市前不补值。"
                   "“完整期”指六项指标均可按旧版规则计算，不等于原始行情完全无缺口。")
        if stats and bool(df.loc[stats["asof"], "近19日含补值"]):
            st.warning("近期行情存在缺口，当前指数按旧版规则沿用前值计算；数据页可查看逐日补值标记。")

    tabs = st.tabs(["📈 综合指数看板", "🔬 历史回测分析", "🔎 主动基金筛选", "🗂️ 数据与备份"])
    with tabs[0]:
        st.subheader("综合指数走势")
        if frame.empty:
            st.info("当前日期范围没有行情，请选择起止日期或调整范围。")
        else:
            st.plotly_chart(plot_index(frame), width="stretch")
        if not show_early and len(ref_df):
            st.caption("查看更早曲线：勾选侧边栏“显示早期参考历史”，并选择“全部可用历史”或指定起止日期。")
        if stats:
            st.subheader("历史分布：完整期与早期参考分开显示")
            hist = go.Figure()
            hist.add_trace(go.Histogram(x=stats["full"], name="完整模型", xbins=dict(start=0,end=100,size=2), marker_color=C_BLUE, opacity=.75))
            early_values = df.loc[:stats["asof"], "早期参考指数"].dropna()
            hist.add_trace(go.Histogram(x=early_values, name="早期参考", xbins=dict(start=0,end=100,size=2), marker_color="#a4a8b5", opacity=.5))
            hist.add_vline(x=stats["value"], line_color="#F7DC6F", annotation_text=f"当前完整指数 {stats['value']:.1f}")
            hist.update_layout(**dark_layout(height=320, y_title="交易日数"))
            hist.update_layout(barmode="overlay", showlegend=True)
            hist.update_xaxes(range=[0,100], title="指数值")
            st.plotly_chart(hist, width="stretch")
        cols = st.columns(2)
        for col,field,color in zip(cols,["综合情绪指标","综合资金指标"],[C_BLUE,"#FF9F0A"]):
            with col:
                st.markdown(f"**{field}**")
                fig = go.Figure()
                for kind,dash in [("完整模型","solid"),("早期参考","dash")]:
                    if kind == "早期参考" and not show_early: continue
                    fig.add_trace(go.Scatter(x=frame.index, y=frame[field].where(frame["数据类型"].eq(kind)),
                        name=kind, line=dict(color=color if kind=="完整模型" else C_MUTED,dash=dash),connectgaps=False))
                fig.update_layout(**dark_layout(height=250))
                st.plotly_chart(fig, width="stretch")
    with tabs[1]:
        render_backtest(df)
    with tabs[2]:
        render_fund_screener()
    with tabs[3]:
        st.subheader("历史覆盖与参与指标")
        st.write("完整指数：" + period_text(df["完整指数"]))
        st.write("早期参考：" + period_text(df["早期参考指数"]))
        coverage = []
        for t in TICKERS:
            series = raw[("Close",t)].dropna()
            coverage.append({"标的": t, "最早行情": str(series.index[0].date()),
                             "最新行情": str(series.index[-1].date()), "有效日线数":len(series)})
        st.dataframe(pd.DataFrame(coverage), hide_index=True, width="stretch")
        st.dataframe(pd.DataFrame([{"指标":k,"含义":v,"首次有效日期":str(df[k].first_valid_index())[:10]}
                                  for k,v in FACTOR_NAMES.items()]), hide_index=True)
        st.caption("覆盖率是当前有效指标的原始权重占比，不是准确率。"
                   "因连续两次平滑，首次六项齐全后还需预热才能进入完整期。"
                   "图表和持有周期以 QQQ 有效交易日为准。")
        st.caption(f"行情处理策略：{USER_CONFIG['MISSING_POLICY']}。legacy 保留原版补值与指标日历；"
                   "strict 不补缺失价格或成交量。切换到 strict 属于数据处理口径变更，会改变部分读数。")
        with st.expander("逐日核对指标与模型类型"):
            st.dataframe(df.tail(500), width="stretch")
            st.caption("页面列出最近 500 个交易日；CSV 包含全部日期及 P1–P6。")
        st.download_button("下载完整指数与参考历史 CSV", df.to_csv().encode("utf-8-sig"),
                           "bubble_history_v31.csv", "text/csv")
        st.download_button("下载原始行情备份 ZIP（可恢复）", backup_bytes(raw),
                           f"bubble_market_{raw.index[-1]:%Y%m%d}.zip", "application/zip")
        st.caption("本页 ZIP 保存泡沫指数底层行情；基金筛选结果请在基金页面另行导出 CSV。")
        st.info("默认在运行服务器的 bubble_data 目录保存独立行情快照。Streamlit Community Cloud "
                "不保证本地文件永久保留，请下载 ZIP 到自己的电脑；可在侧边栏上传恢复。"
                "本版本尚未连接外部云存储，也不会在 App 关闭后定时下载。")
        st.caption("每次更新保存整份一致的复权行情，避免新旧复权价格直接拼接。"
                   "旧快照不会被新的下载覆盖；未缩短任何计算窗口。")


def render_backtest(df):
    st.subheader("历史回测：仅使用完整模型的信号")
    full = df["完整指数"].dropna()
    if full.empty:
        st.info("暂无完整模型历史，不能开展正式回测。")
        return
    st.caption(period_text(full) + "。早期参考段不参与；逐日信号存在重叠，样本并非独立交易。")
    results = run_backtest(df)
    rows=[]
    for r in results:
        row={"区间":r["区间"], "信号天数":r["信号天数"]}
        for p,_ in PERIODS:
            row[f"{p}均收益(%)"] = r[f"{p}_avg"]
            row[f"{p}胜率(%)"] = r[f"{p}_win"]
            row[f"{p}有效样本"] = r[f"{p}_n"]
        rows.append(row)
    st.dataframe(pd.DataFrame(rows).round(2), hide_index=True, width="stretch")
    period=st.selectbox("持有周期",[p for p,_ in PERIODS],index=2)
    fig=go.Figure()
    fig.add_trace(go.Bar(x=[r["区间"] for r in results],y=[r[f"{period}_avg"] for r in results],name="均值",marker_color=C_BLUE))
    fig.add_trace(go.Scatter(x=[r["区间"] for r in results],y=[r[f"{period}_median"] for r in results],name="中位数",mode="markers",marker=dict(color="#F7DC6F",size=12,symbol="diamond")))
    fig.update_layout(**dark_layout(height=400,y_title="收益率 (%)"))
    fig.update_layout(showlegend=True)
    st.plotly_chart(fig,width="stretch")
    win=go.Figure(go.Bar(x=[r["区间"] for r in results],y=[r[f"{period}_win"] for r in results],marker_color=C_BLUE))
    win.add_hline(y=50,line_dash="dash",line_color="#FF9F0A")
    win.update_layout(**dark_layout(height=320,y_range=[0,100],y_title="正收益胜率 (%)"))
    st.plotly_chart(win,width="stretch")
    multi=go.Figure()
    for p,_ in PERIODS:
        multi.add_trace(go.Scatter(x=[r["区间"] for r in results],y=[r[f"{p}_avg"] for r in results],name=p,mode="lines+markers"))
    multi.update_layout(**dark_layout(height=360,y_title="平均收益率 (%)"))
    multi.update_layout(showlegend=True)
    st.plotly_chart(multi,width="stretch")
    st.caption("采用当日收盘指数与收盘价的条件收益统计，未模拟可执行成交、成本或资金管理；历史收益不代表未来表现。")


# ============================================================
# 主动基金筛选 / Active fund screening (independent of bubble model)
# ============================================================
import re
from concurrent.futures import ThreadPoolExecutor, as_completed

BENCHMARKS = {
    "纳斯达克100": {"index": "^NDX", "proxy": "QQQ"},
    "费城半导体": {"index": "^SOX", "proxy": "SOXQ"},
    "标普500": {"index": "^GSPC", "proxy": "SPY"},
}
ACTIVE_TYPES = {
    "QDII-普通股票", "QDII-混合偏股", "QDII-混合平衡", "QDII-混合灵活",
    "股票型", "混合型-偏股", "混合型-平衡", "混合型-灵活",
}
PUBLIC_HEADERS = {"User-Agent": "Mozilla/5.0", "Referer": "https://fund.eastmoney.com/"}


def select_chart_frame(df, include_early, mode, count=None, start=None, end=None):
    """Chart filters never change percentile/backtest populations / 只裁图表。"""
    column = "总泡沫指数" if include_early else "完整指数"
    valid = df[column].dropna()
    if valid.empty:
        return df.iloc[:0].copy()
    frame = df.loc[valid.index[0]:].copy()
    if mode == "输入交易日数":
        frame = frame.tail(int(count))
    elif mode == "指定起止日期":
        if start is None or end is None or pd.Timestamp(start) > pd.Timestamp(end):
            raise ValueError("开始日期不能晚于结束日期。")
        frame = frame.loc[pd.Timestamp(start):pd.Timestamp(end)]
    if not include_early:
        frame["早期参考指数"] = np.nan
    return frame


def render_time_controls(df, include_early):
    column = "总泡沫指数" if include_early else "完整指数"
    valid = df[column].dropna()
    if valid.empty:
        st.sidebar.info("该口径暂无曲线；可勾选早期参考历史。")
        return df.iloc[:0].copy()
    low, high = valid.index[0].date(), df.index[-1].date()
    maximum = len(df.loc[valid.index[0]:])
    minimum = min(100, maximum)
    mode = st.sidebar.radio("时间选择方式", ["输入交易日数", "指定起止日期", "全部可用历史"], key="chart_mode")
    st.sidebar.caption(f"可用日期：{low} 至 {high}；{maximum:,} 个交易日。")
    if mode == "输入交易日数":
        # A scope change can reduce max_value; use a bounded, scope-specific widget.
        count = st.sidebar.number_input(f"最近交易日数（{minimum}–{maximum}）", min_value=minimum,
                max_value=maximum, value=min(400,maximum), step=1, key=f"chart_days_{include_early}_{maximum}")
        return select_chart_frame(df, include_early, mode, count=count)
    if mode == "指定起止日期":
        chosen = st.sidebar.date_input("起止日期（含首尾）", value=(max(low, (pd.Timestamp(high)-pd.DateOffset(years=1)).date()),high),
                min_value=low, max_value=high, format="YYYY-MM-DD", key=f"chart_dates_{include_early}_{low}_{high}")
        if len(chosen) != 2:
            st.sidebar.info("请选择结束日期。")
            return df.iloc[:0].copy()
        return select_chart_frame(df, include_early, mode, start=chosen[0], end=chosen[1])
    return select_chart_frame(df, include_early, mode)


def parse_fund_catalog(text):
    match = re.search(r"\bvar\s+r\s*=\s*", text)
    if not match:
        raise ValueError("基金目录格式变化，未找到公开目录数组。")
    rows, _ = json.JSONDecoder().raw_decode(text[match.end():].lstrip())
    frame = pd.DataFrame(rows, columns=["基金代码", "拼音缩写", "基金名称", "基金类型", "拼音全称"])
    if frame.empty or frame["基金代码"].duplicated().any():
        raise ValueError("基金目录为空或含重复代码。")
    return frame[["基金代码","基金名称","基金类型"]]


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fund_catalog():
    response = requests.get("https://fund.eastmoney.com/js/fundcode_search.js", headers=PUBLIC_HEADERS, timeout=20)
    response.raise_for_status()
    response.encoding = "utf-8-sig"
    return parse_fund_catalog(response.text)


def active_candidates(catalog):
    # Public name/type screening is a candidate filter, not prospectus certification.
    passive = catalog["基金名称"].str.contains(r"指数|ETF|联接|标普|纳斯达克100|纳指100", case=False, na=False)
    other = catalog["基金名称"].str.contains(r"美元|港币|后端|FOF|REIT", case=False, na=False)
    return catalog.loc[catalog["基金类型"].isin(ACTIVE_TYPES) & ~passive & ~other].copy()


def parse_fund_nav(text, expected_code):
    """Read provider JSON as data, never execute downloaded JavaScript.
    使用公布日增长率复利连乘；不直接把累计净值比当成总收益。
    """
    code_match = re.search(r'\bvar\s+fS_code\s*=\s*"(\d{6})"', text)
    match = re.search(r"\bvar\s+Data_netWorthTrend\s*=\s*", text)
    if not code_match or code_match.group(1) != expected_code or not match:
        raise ValueError("净值响应代码不匹配，或缺少日净值序列。")
    rows, _ = json.JSONDecoder().raw_decode(text[match.end():].lstrip())
    nav = pd.DataFrame(rows)
    if not {"x","y","equityReturn"}.issubset(nav.columns) or nav.empty:
        raise ValueError("缺少日期、单位净值或公布日增长率。")
    dates = pd.to_datetime(nav["x"],unit="ms",utc=True).dt.tz_convert("Asia/Shanghai").dt.tz_localize(None).dt.normalize()
    frame = pd.DataFrame({"nav":pd.to_numeric(nav["y"],errors="coerce").to_numpy(),
                          "return":pd.to_numeric(nav["equityReturn"],errors="coerce").to_numpy()/100}, index=pd.DatetimeIndex(dates))
    frame = frame.sort_index()
    if frame.index.has_duplicates:
        raise ValueError("净值日期重复，不能直接合并。")
    frame["return"] = frame["return"].where(np.isfinite(frame["return"]) & frame["return"].gt(-1))
    frame["nav"] = frame["nav"].where(np.isfinite(frame["nav"]) & frame["nav"].gt(0))
    return frame


@st.cache_data(ttl=21600, show_spinner=False)
def fetch_fund_nav(code):
    if not re.fullmatch(r"\d{6}", code):
        raise ValueError("基金代码必须是六位数字。")
    response = requests.get(f"https://fund.eastmoney.com/pingzhongdata/{code}.js", headers=PUBLIC_HEADERS, timeout=20)
    response.raise_for_status()
    response.encoding="utf-8-sig"
    return parse_fund_nav(response.text, code)


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_screen_benchmarks(start_date, end_date, kind, currency):
    tickers = [v["proxy" if kind=="ETF复权收益代理" else "index"] for v in BENCHMARKS.values()]
    if currency=="人民币": tickers += ["CNY=X"]
    raw = yf.download(tickers, start=start_date, end=end_date, auto_adjust=True,
                      progress=False, threads=True, interval="1d")
    if raw is None or raw.empty or "Close" not in raw:
        raise ValueError("未获得基准行情。")
    close=raw["Close"].copy()
    close.index=pd.to_datetime(close.index).tz_localize(None).normalize()
    close=close.loc[~close.index.duplicated()].sort_index()
    us_today=pd.Timestamp.now(tz="America/New_York").date()
    close=close.loc[close.index.date<us_today]
    result={}
    for name, symbols in BENCHMARKS.items():
        symbol=symbols["proxy" if kind=="ETF复权收益代理" else "index"]
        if symbol not in close or close[symbol].dropna().empty:
            raise ValueError(f"{name} ({symbol}) 行情缺失，未用其他指数替代。")
        s=close[symbol].where(close[symbol]>0).dropna()
        if currency=="人民币":
            if "CNY=X" not in close or close["CNY=X"].dropna().empty:
                raise ValueError("缺少 USD/CNY 汇率，不能静默改用美元口径。")
            fx=close["CNY=X"].where(close["CNY=X"]>0).dropna()
            # Backward/as-of only, no future FX information; tolerate short holidays.
            fx=fx.reindex(s.index,method="ffill",tolerance=pd.Timedelta(days=4))
            s=s*fx
        result[name]=s
    frame=pd.DataFrame(result)
    if frame.dropna().empty:
        raise ValueError("基准与汇率没有共同有效区间。")
    return frame


def fund_wealth(nav, start, end):
    window=nav.loc[start:end]
    if len(window)<2 or window["nav"].isna().any() or window["return"].iloc[1:].isna().any():
        raise ValueError("比较区间内净值或公布增长率缺失，未以 0 收益填充。")
    growth=window["return"].copy()
    growth.iloc[0]=0.0
    wealth=(1+growth).cumprod()
    if not np.isfinite(wealth).all() or (wealth<=0).any():
        raise ValueError("净值复利路径无效。")
    return wealth


def score_pair(fund, benchmark, bonus_max=40.0, block_size=20):
    """Transparent descriptive score, not an estimated chance of future success.
    所有输入必须是同一组日期上的财富路径；不优化时滞或挑选最优窗口。
    """
    pair=pd.concat([fund.rename("fund"),benchmark.rename("benchmark")],axis=1)
    if pair.isna().any().any() or len(pair)<31 or (pair<=0).any().any():
        raise ValueError("至少需要 30 个完整匹配收益区间。")
    level=pair/pair.iloc[0]
    ret=level.pct_change(fill_method=None).iloc[1:]
    if ret.fund.std()<1e-10 or ret.benchmark.std()<1e-10:
        raise ValueError("路径近乎不变，无法可靠计算相关性。")
    corr=float(ret.fund.corr(ret.benchmark))
    fund_return=float(level.fund.iloc[-1]-1)
    index_return=float(level.benchmark.iloc[-1]-1)
    excess=fund_return-index_return
    rmse=float(np.sqrt(np.mean((level.fund-level.benchmark)**2)))
    # Heuristic decay scales are 10 percentage points; edit the following formula to change them.
    similarity=100*(.55*max(corr,0)+.25*np.exp(-rmse/.10)+.20*np.exp(-abs(excess)/.10))
    block_excess=[]
    for begin in range(0,len(level)-block_size,block_size):
        finish=begin+block_size
        change=level.iloc[finish]/level.iloc[begin]-1
        block_excess.append(float(change.fund-change.benchmark))
    win=float(np.mean(np.array(block_excess)>0)) if block_excess else np.nan
    # No bonus for one lucky final spike, negative excess, or insufficient blocks.
    bonus=0.0
    if len(block_excess)>=3 and excess>0 and corr>=.6 and win>.5:
        bonus=float(bonus_max*(1-np.exp(-excess/.10))*((win-.5)/.5)*max(corr,0))
    beta=float(ret.fund.cov(ret.benchmark)/ret.benchmark.var())
    drawdown=float((level.fund/level.fund.cummax()-1).min())
    return {"综合得分":float(similarity+bonus),"相似度分":float(similarity),"稳定超额加分":bonus,
            "收益相关系数":corr,"基金收益(%)":fund_return*100,"基准收益(%)":index_return*100,
            "超额收益(百分点)":excess*100,"路径偏差(百分点)":rmse*100,
            "分段跑赢比例(%)":win*100,"完整分段数":len(block_excess),"Beta":beta,"最大回撤(%)":drawdown*100}


def compare_funds(navs, benchmarks, lookback, bonus_max=40, lag=0, maximum_stale_days=10):
    """One common time grid for all funds and benchmarks; excludes short/stale series.
    中美节假日通过共同日期聚合收益，不为节假日补造零收益。
    """
    if lag not in (0,1): raise ValueError("时滞只能是 0 或 1，且在评分前固定。")
    base=benchmarks.dropna().sort_index()
    if lag:
        base=base.shift(lag).dropna()
    if len(base)<lookback+1:
        raise ValueError(f"基准仅有 {len(base)-1} 个收益区间，不足请求的 {lookback}。")
    requested=base.tail(lookback+1)
    start,end=requested.index[0],requested.index[-1]
    usable={}; errors=[]
    for code,nav in navs.items():
        try:
            if nav.empty or nav.index[0]>start:
                raise ValueError("成立/数据时间太短，无法覆盖请求起点")
            if (end-nav.index[-1]).days>maximum_stale_days:
                raise ValueError("净值过旧，超过允许滞后天数")
            wealth=fund_wealth(nav,start-pd.Timedelta(days=10),end)
            common=wealth.index.intersection(requested.index)
            if len(common)<max(31,int(.70*len(requested))):
                raise ValueError("有效日期不足请求窗口的 70%")
            usable[code]=wealth
        except ValueError as exc: errors.append({"基金代码":code,"原因":str(exc)})
    if not usable: return pd.DataFrame(),{},pd.DataFrame(errors),{}
    common=requested.index
    for wealth in usable.values(): common=common.intersection(wealth.index)
    common=common.sort_values()
    if len(common)<max(31,int(.65*len(requested))):
        raise ValueError("候选基金共同日期过少。请减少异常/缺失较多的基金后重试；未自动缩短各自比较周期。")
    if (common[0]-start).days>10 or (end-common[-1]).days>maximum_stale_days:
        raise ValueError("共同区间偏离请求起止日期过多，未进行不同时段排名。")
    records=[]; paths={}
    for code,wealth in usable.items():
        f=wealth.reindex(common)
        paths[code]=f/f.iloc[0]
        for name in base.columns:
            try:
                result=score_pair(f,base.loc[common,name],bonus_max=bonus_max)
                records.append({"基金代码":code,"比较基准":name,**result})
            except ValueError as exc: errors.append({"基金代码":code,"原因":f"{name}: {exc}"})
    for name in base.columns:
        b=base.loc[common,name]
        paths[name]=b/b.iloc[0]
    info={"请求开始":str(start.date()),"请求结束":str(end.date()),
          "实际开始":str(common[0].date()),"实际结束":str(common[-1].date()),
          "共同收益区间数":len(common)-1,"候选有效数":len(usable),"固定时滞":lag}
    return pd.DataFrame(records),paths,pd.DataFrame(errors),info


def render_fund_screener():
    st.subheader("主动基金 · 纳指100 / 费半 / 标普500 相似度筛选")
    st.caption("按历史相似度与持续超额收益排序，基金在支付宝的上架及额度由你最终确认。"
               "本模块独立于泡沫指数，不改变原模型或百分位。")
    st.caption("净值来自天天基金公开数据；本榜不代表支付宝在售清单，也不核验实时限额。")
    if st.button("加载 / 刷新主动基金目录",key="load_fund_catalog"):
        fetch_fund_catalog.clear()
        try: st.session_state["fund_catalog"]=fetch_fund_catalog()
        except Exception as exc: st.error(f"基金目录获取失败：{exc}")
    catalog=st.session_state.get("fund_catalog")
    if catalog is None:
        st.caption("点击上方加载目录，再筛选候选基金或输入自己的代码。不会自动下载全部基金净值。")
        return
    candidates=active_candidates(catalog)
    with st.expander("主动基金候选目录（不代表支付宝在售）",expanded=True):
        scope=st.radio("候选范围",["主动 QDII","主动股票 / 混合（含境内）"],horizontal=True,key="fund_scope")
        subset=candidates[candidates["基金类型"].str.startswith("QDII")] if scope=="主动 QDII" else candidates
        search=st.text_input("搜索基金名称或代码",key="fund_search")
        if search.strip(): subset=subset[subset["基金名称"].str.contains(search.strip(),regex=False)|subset["基金代码"].str.contains(search.strip(),regex=False)]
        st.caption(f"符合条件 {len(subset):,} 个份额。已排除名称/类型明确为指数、ETF、联接及非人民币份额；"
                   "A/C 份额分别计分，不应视为不同投资策略。主动属性最终以基金说明书为准。")
        all_candidates=st.checkbox("比较当前筛选目录的全部基金",value=False,key="all_fund_candidates")
        selected=st.multiselect("加入比较的基金",subset["基金代码"].tolist(),
                   format_func=lambda c:f"{c} · {catalog.set_index('基金代码').loc[c,'基金名称']}",key="selected_funds")
        st.dataframe(subset,hide_index=True,width="stretch",height=220)
    text=st.text_area("补充基金代码（六位，逗号或换行分隔）",key="fund_codes",placeholder="填写你关注的主动基金代码")
    if all_candidates: selected=subset["基金代码"].tolist()
    codes=list(dict.fromkeys(selected+re.findall(r"(?<!\d)\d{6}(?!\d)",text)))
    if not codes:
        st.caption("先从目录选择基金、选择全部当前候选，或输入代码。")
        return
    allowed=set(candidates["基金代码"])
    rejected=[c for c in codes if c not in allowed]
    if rejected: st.warning("以下代码未通过主动人民币份额筛选，已排除："+", ".join(rejected))
    codes=[c for c in codes if c in allowed]
    if not codes: return
    if len(codes)>300:
        st.error("单次最多比较 300 个份额，请缩小候选范围。")
        return
    st.caption(f"本次候选 {len(codes)} 个份额；点击计算后才获取其净值。")
    c1,c2,c3=st.columns(3)
    lookback=int(c1.number_input("回看美股交易日数",min_value=60,max_value=1500,value=200,step=1,key="fund_lookback"))
    end_date=c2.date_input("筛选截止日期",value=pd.Timestamp.now(tz="Asia/Shanghai").date(), max_value=pd.Timestamp.now(tz="Asia/Shanghai").date(),key="fund_end_date")
    bonus=float(c3.number_input("稳定超额最高加分",min_value=0.0,max_value=50.0,value=40.0,step=1.0,key="fund_bonus"))
    c1,c2,c3=st.columns(3)
    kind=c1.selectbox("比较口径",["ETF复权收益代理","原始价格指数"],key="fund_kind")
    currency=c2.selectbox("比较币种",["人民币","美元指数对人民币基金（未校正）"],key="fund_currency")
    lag=c3.selectbox("基金日期对应美股日期",[0,1],format_func=lambda x:"同一日期（QDII 默认）" if x==0 else "前一个美股交易日",key="fund_lag")
    if kind=="ETF复权收益代理":
        st.caption("基准分别为 QQQ、SOXQ、SPY 的复权行情，近似包含分红再投资及 ETF 费用；"
                   "是对应指数的收益代理，不是指数本体。")
    else:
        st.warning("NDX、SOX、GSPC 是价格指数，不含股息再投资；基金公布增长率与其收益口径不同，超额不能直接解释为管理能力。")
    if currency=="人民币": st.caption("人民币基准≈美元基准×USD/CNY 市场汇率；不代表基金使用的精确估值汇率或对冲结果。")
    else: st.warning("未统一币种，汇率变化会影响收益差与排名。")
    included=codes
    signature=hashlib.sha256(json.dumps({"codes":included,"lookback":lookback,"kind":kind,"currency":currency,"lag":lag,"bonus":bonus,"end":str(end_date)},sort_keys=True).encode()).hexdigest()
    if st.button("计算相似度与排名",type="primary",key="run_fund_screen"):
        if not included:
            st.warning("请先选择基金。")
        else:
            try:
                end=end_date+pd.Timedelta(days=1)
                begin=end-pd.Timedelta(days=int(lookback*1.9)+90)
                with st.spinner("获取指数与汇率…"):
                    benchmarks=fetch_screen_benchmarks(str(begin),str(end),kind,"人民币" if currency=="人民币" else "美元")
                navs={}; failures=[]
                bar=st.progress(0.0,text="获取基金净值…")
                with ThreadPoolExecutor(max_workers=3) as executor:
                    jobs={executor.submit(fetch_fund_nav,c):c for c in included}
                    for i,job in enumerate(as_completed(jobs),1):
                        code=jobs[job]
                        try: navs[code]=job.result()
                        except Exception as exc: failures.append({"基金代码":code,"原因":f"净值获取失败：{exc}"})
                        bar.progress(i/len(jobs),text=f"已处理 {i}/{len(jobs)} 个份额")
                ranking,paths,errors,info=compare_funds(navs,benchmarks,lookback,bonus,lag)
                errors=pd.concat([errors,pd.DataFrame(failures)],ignore_index=True)
                st.session_state["fund_screen_result"]={"signature":signature,"ranking":ranking,"paths":paths,"errors":errors,"info":info,
                    "computed_at":pd.Timestamp.now(tz="Asia/Shanghai").isoformat(),"kind":kind,"currency":currency,"bonus":bonus}
            except Exception as exc: st.error(f"筛选未完成：{exc}")
    result=st.session_state.get("fund_screen_result")
    if not result: return
    if result["signature"]!=signature:
        st.info("候选或计算参数已变化，请重新计算；不展示旧参数下的排名。")
        return
    ranking=result["ranking"]
    if not result["errors"].empty:
        with st.expander("未参与评分的基金及原因",expanded=ranking.empty):
            st.dataframe(result["errors"],hide_index=True,width="stretch")
    if ranking.empty:
        st.warning("没有符合数据要求的评分结果。")
        return
    info=result["info"]
    st.caption(f"请求 {info['请求开始']} 至 {info['请求结束']}；共同有效日期 {info['实际开始']} 至 {info['实际结束']}，"
               f"共 {info['共同收益区间数']} 个收益区间。所有基金与基准使用相同日期，长假期间收益按实际间隔累计。")
    st.caption("计算时间："+result["computed_at"]+"；这是当前候选集合的历史描述性排名，不是全市场排名或未来收益预测。")
    chosen=st.selectbox("查看哪个基准的排名",list(BENCHMARKS),key="rank_benchmark")
    minimum_corr=st.slider("最低收益相关系数",min_value=0.0,max_value=1.0,value=.6,step=.05,key="fund_corr")
    ranking=ranking.merge(catalog,on="基金代码",how="left",validate="many_to_one")
    ranking=ranking[(ranking["比较基准"]==chosen)&(ranking["收益相关系数"]>=minimum_corr)].sort_values("综合得分",ascending=False)
    display_cols=["基金代码","基金名称","综合得分","相似度分","稳定超额加分","收益相关系数",
                  "基金收益(%)","基准收益(%)","超额收益(百分点)","分段跑赢比例(%)","完整分段数","最大回撤(%)","Beta"]
    st.markdown("**主动基金相似度排名**")
    if ranking.empty: st.info("没有满足最低相关系数的基金，可调整阈值或候选范围。")
    else: st.dataframe(ranking[display_cols].round(3),hide_index=True,width="stretch")
    if not ranking.empty:
        options=ranking["基金代码"].tolist()
        selected_plot=st.multiselect("叠加收益曲线",options,default=options[:3],key="fund_plot_codes")
        figure=go.Figure()
        b=result["paths"][chosen]
        figure.add_trace(go.Scatter(x=b.index,y=(b-1)*100,name=chosen,line=dict(color="#F7DC6F",width=3)))
        for code in selected_plot:
            path=result["paths"][code]
            figure.add_trace(go.Scatter(x=path.index,y=(path-1)*100,name=f"{code} {catalog.set_index('基金代码').loc[code,'基金名称']}"))
        figure.update_layout(**dark_layout(height=450,y_title="共同起点累计收益 (%)"))
        figure.update_layout(showlegend=True,legend=dict(orientation="h",y=-.2))
        st.plotly_chart(figure,width="stretch")
        export=ranking[display_cols].copy()
        export["口径"]=kind; export["币种"]=currency
        export["实际开始"]=info["实际开始"]; export["实际结束"]=info["实际结束"]
        export["共同收益区间数"]=info["共同收益区间数"]; export["计算时间"]=result["computed_at"]
        export["支付宝状态"]="上架及实时额度未核验，请在支付宝自行确认"
        st.download_button("下载当前基准筛选结果 CSV",export.to_csv(index=False).encode("utf-8-sig"),"fund_similarity_ranking.csv","text/csv")
    with st.expander("评分方法与数据来源"):
        st.markdown("相似度满分 **100**：55% 收益相关性＋25% 累计路径接近度＋20% 最终收益接近度。"
                    "路径与最终收益差按 10 个百分点尺度指数衰减，负相关不加相关性分。")
        st.markdown(f"稳定超额最多加 **{bonus:g}** 分：总超额为正、相关系数至少 0.6，且至少三个互不重叠的 "
                    "20 区间分段中，超过一半跑赢，才按超额幅度和持续性加分。综合得分可超过 100；分数不是成功概率。")
        st.caption("基金使用公布的日增长率复利连乘，避免把未复权单位净值或累计净值比值误当总收益。"
                   "未扣个人申购赎回费。回撤按共同采样日期估计，可能低估日期间回撤。"
                   "币种、净值日期解释和固定时滞需结合基金估值说明核对；未搜索最优时滞。"
                   "候选来自当前存续目录，存在存续偏差；稳定超额不是风险调整后的 Alpha。")
        st.markdown("[天天基金公开目录](https://fund.eastmoney.com/) · "
                    "[Yahoo Finance](https://finance.yahoo.com/) · "
                    "[QQQ](https://www.invesco.com/qqq-etf/en/home.html) · "
                    "[SOXQ](https://www.invesco.com/us/en/financial-products/etfs/invesco-phlx-semiconductor-etf.html) · "
                    "[SPY](https://www.ssga.com/us/en/individual/etfs/state-street-spdr-sp-500-etf-trust-spy)")


C_BG       = "#131722"
C_PANEL    = "#1e222d"
C_BORDER   = "#2a2e39"
C_BLUE     = "#2962ff"
C_TEXT     = "#d1d4dc"
C_MUTED    = "#787b86"

ZONES = [
    (0,   33,  "💎 极限大底/重仓 (<33)",  "#00FFFF"), 
    (33,  39,  "🟩 大幅加仓机会 (33-39)", "#32CD32"),
    (39,  41.5,"🟢 优质定投区域 (39-41.5)","#90EE90"),
    (41.5,45,  "🟡 恐慌底部分界 (41.5-45)","#FFD700"),
    (45,  52,  "📉 趋势变坏/下行 (45-52)", "#FF3B30"),
    (52,  60,  "⚠️ 高位背离警告 (52-60)", "#FF9F0A"),
    (60,  75,  "🔵 合理价位/持有 (60-75)", "#2962ff"),
    (75,  101, "🚨 估值偏高/警惕 (>75)",   "#F7DC6F"),
]
PERIODS = [("1个月", 21), ("3个月", 63), ("6个月", 126), ("1年", 252)]

def dark_layout(height=520, y_range=None, y_title=None, title_text=None):
    """返回统一的深色 Plotly 布局字典"""
    layout = dict(
        height=height,
        paper_bgcolor=C_BG,
        plot_bgcolor=C_PANEL,
        font=dict(color=C_TEXT, family="Courier New, monospace", size=15),
        margin=dict(l=8, r=8, t=30 if title_text else 10, b=8),
        hovermode="x unified",
        hoverlabel=dict(font_size=15, font_family="Courier New, monospace"),
        showlegend=False,
        xaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, showgrid=True, tickfont=dict(color=C_MUTED, size=14)),
        yaxis=dict(gridcolor=C_BORDER, linecolor=C_BORDER, showgrid=True, tickfont=dict(color=C_MUTED, size=14)),
    )
    if y_range:
        layout["yaxis"]["range"] = y_range
    if y_title:
        layout["yaxis"]["title"] = dict(text=y_title, font=dict(color=C_MUTED, size=15))
    if title_text:
        layout["title"] = dict(text=title_text, font=dict(color=C_TEXT, size=16), x=0, xanchor="left")
    return layout


if __name__ == "__main__":
    st.set_page_config(page_title="AI泡沫指数 V3.2", page_icon="📈", layout="wide")
    
    # ============================================================
    # Bloomberg / TradingView 深色主题 CSS
    # ============================================================
    st.markdown("""
    <style>
    /* 主背景 */
    .stApp { background-color: #131722; color: #d1d4dc; }
    
    /* 侧边栏 */
    section[data-testid="stSidebar"] {
        background-color: #1e222d;
        border-right: 1px solid #2a2e39;
    }
    section[data-testid="stSidebar"] * { color: #d1d4dc !important; }
    
    /* 指标卡片 */
    [data-testid="metric-container"] {
        background-color: #1e222d;
        border: 1px solid #2a2e39;
        border-radius: 8px;
        padding: 16px 20px;
    }
    [data-testid="stMetricValue"] {
        color: #d1d4dc !important;
        font-family: 'Courier New', monospace !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
    }
    [data-testid="stMetricLabel"] { color: #787b86 !important; font-size: 0.9rem !important; font-weight: 600 !important; }
    [data-testid="stMetricDelta"] { font-family: 'Courier New', monospace !important; font-weight: 600 !important; }
    
    /* 标题 */
    h1, h2, h3 { color: #d1d4dc !important; }
    h1 { font-family: 'Courier New', monospace !important; border-bottom: 1px solid #2a2e39; padding-bottom: 8px; }
    
    /* 分割线 */
    hr { border-color: #2a2e39 !important; }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1e222d;
        border-bottom: 1px solid #2a2e39;
        gap: 4px;
    }
    .stTabs [data-baseweb="tab"] {
        color: #787b86;
        background-color: transparent;
        border-radius: 6px 6px 0 0;
        padding: 8px 20px;
        font-family: 'Courier New', monospace;
    }
    .stTabs [aria-selected="true"] {
        color: #2962ff !important;
        border-bottom: 2px solid #2962ff !important;
        background-color: rgba(41,98,255,0.08) !important;
    }
    
    /* Selectbox / Dropdown */
    [data-baseweb="select"] { background-color: #1e222d !important; border-color: #2a2e39 !important; }
    [data-baseweb="select"] * { color: #d1d4dc !important; background-color: #1e222d !important; }
    
    /* Dataframe */
    [data-testid="stDataFrame"] { border: 1px solid #2a2e39; border-radius: 6px; }
    
    /* Spinner */
    .stSpinner > div { border-top-color: #2962ff !important; }
    
    /* Slider */
    [data-testid="stSlider"] [data-baseweb="slider"] [role="slider"] {
        background-color: #2962ff !important;
        border-color: #2962ff !important;
    }
    
    /* Markdown */
    .stMarkdown, p { color: #d1d4dc !important; }
    </style>
    """, unsafe_allow_html=True)
    
    # ============================================================
    # 设计常量
    # ============================================================
    render_main()
