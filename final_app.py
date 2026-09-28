"""AI bubble dashboard V3.3 / 境内与QDII主动基金的三个独立榜单。"""
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
MODEL_VERSION = "3.3-monthly-universe-scan"
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
    st.markdown("# 🛡️ 私人量化终端：AI 泡沫综合指数 V3.3")
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
FUND_SCOPES = ["全部主动基金（境内 + QDII）", "仅境内股票 / 混合（非QDII）", "仅主动 QDII"]
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


def filter_fund_scope(frame, scope):
    """Fund category is only an optional filter, never a benchmark assignment.
    基金类别不决定入哪个指数榜；所有候选均分别对比三个基准。
    """
    qdii = frame["基金类型"].str.startswith("QDII", na=False)
    if scope == FUND_SCOPES[1]:
        return frame.loc[~qdii].copy()
    if scope == FUND_SCOPES[2]:
        return frame.loc[qdii].copy()
    return frame.copy()


def candidate_batch(codes, batch_number, batch_size=300):
    """Deterministic coverage of a large universe without silently taking top N.
    大候选池按代码稳定分批，界面明确本批范围，不冒充全市场排名。
    """
    ordered = sorted(set(codes))
    total = max(1, (len(ordered) + batch_size - 1) // batch_size)
    if not 1 <= batch_number <= total:
        raise ValueError("批次超出候选范围。")
    start = (batch_number - 1) * batch_size
    return ordered[start:start + batch_size], total


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


def benchmark_ranking(ranking, benchmark, minimum_corr):
    """Independent board: never pool, average or blend different benchmarks.
    每个榜单仅按该指数的评分排序，同一基金可以进入多个榜单。
    """
    selected = ranking.loc[(ranking["比较基准"] == benchmark)
                           & (ranking["收益相关系数"] >= minimum_corr)].copy()
    selected = selected.sort_values(["综合得分", "基金代码"], ascending=[False, True]).reset_index(drop=True)
    selected.insert(0, "排名", np.arange(1, len(selected) + 1))
    return selected


def render_benchmark_board(ranking, chosen, result, catalog, kind, currency):
    st.markdown(f"**与{chosen}相似的主动基金**")
    st.caption(f"本页全部分数、超额收益与曲线只对比{chosen}，不混入其他指数的评分。")
    minimum_corr = st.slider(f"{chosen} · 最低收益相关系数", min_value=0.0, max_value=1.0,
                            value=.6, step=.05, key=f"fund_corr_{chosen}")
    scope = st.selectbox(f"{chosen} · 显示基金类别", FUND_SCOPES, key=f"board_scope_{chosen}")
    board = benchmark_ranking(filter_fund_scope(ranking, scope), chosen, minimum_corr)
    display_cols = ["排名", "基金代码", "基金名称", "基金类别", "基金类型", "综合得分", "相似度分", "稳定超额加分", "收益相关系数",
                    "基金收益(%)", "基准收益(%)", "超额收益(百分点)", "分段跑赢比例(%)", "完整分段数", "最大回撤(%)", "Beta"]
    if board.empty:
        st.info(f"没有满足{chosen}相关系数门槛的基金，可调整本页阈值或候选范围。")
        return
    display = board[display_cols].rename(columns={"综合得分": "本指数得分", "基准收益(%)": f"{chosen}收益(%)"})
    st.dataframe(display.round(3), hide_index=True, width="stretch")
    options = board["基金代码"].tolist()
    plot_key = f"fund_plot_codes_{chosen}"
    selection_defaults = {}
    if plot_key in st.session_state:
        kept = [c for c in st.session_state[plot_key] if c in options]
        if kept != st.session_state[plot_key]:
            st.session_state[plot_key] = kept
    else:
        selection_defaults["default"] = options[:3]
    selected_plot = st.multiselect(f"{chosen} · 叠加基金曲线", options, key=plot_key, **selection_defaults)
    figure = go.Figure()
    b = result["paths"][chosen]
    figure.add_trace(go.Scatter(x=b.index, y=(b-1)*100, name=chosen, line=dict(color="#F7DC6F", width=3)))
    names = catalog.set_index("基金代码")["基金名称"]
    for code in selected_plot:
        path = result["paths"][code]
        figure.add_trace(go.Scatter(x=path.index, y=(path-1)*100, name=f"{code} {names.loc[code]}"))
    figure.update_layout(**dark_layout(height=450, y_title="共同起点累计收益 (%)"))
    figure.update_layout(showlegend=True, legend=dict(orientation="h", y=-.2))
    st.plotly_chart(figure, width="stretch", key=f"fund_chart_{chosen}")
    info = result["info"]
    export = display.copy()
    export["比较基准"] = chosen
    export["口径"] = kind
    export["币种"] = currency
    export["实际开始"] = info["实际开始"]
    export["实际结束"] = info["实际结束"]
    export["共同收益区间数"] = info["共同收益区间数"]
    export["计算时间"] = result["computed_at"]
    export["支付宝状态"] = "上架及实时额度未核验，请在支付宝自行确认"
    st.download_button(f"下载{chosen}榜单 CSV", export.to_csv(index=False).encode("utf-8-sig"),
                       f"fund_ranking_{BENCHMARKS[chosen]['proxy']}.csv", "text/csv", key=f"fund_export_{chosen}")


# Monthly universe scan / 月度全量扫描。Only standard-library persistence is used.
import sqlite3
import time
import threading
import uuid
from concurrent.futures import wait, FIRST_COMPLETED

FUND_STORE = Path(__file__).resolve().parent / "fund_data"
SCAN_SCHEMA = 1
SCAN_DEFAULTS = {"lookback": 200, "bonus": 40.0, "lag": 0,
                 "kind": "ETF复权收益代理", "currency": "人民币", "publication_buffer": 2,
                 "pool_min_corr": .6, "workers": 6, "auto": True}
_nav_local = threading.local()


def atomic_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    tmp.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False), encoding="utf-8")
    os.replace(tmp, path)


def read_fund_state():
    path = FUND_STORE / "state.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def pack_series(series):
    return {"dates": series.index.strftime("%Y-%m-%d").tolist(), "values": series.astype(float).tolist()}


def unpack_series(data):
    s = pd.Series(data["values"], index=pd.to_datetime(data["dates"]), dtype=float)
    if s.index.has_duplicates or not s.index.is_monotonic_increasing or not np.isfinite(s).all() or (s <= 0).any():
        raise ValueError("备份曲线的日期或数值无效。")
    return s


@st.cache_data(ttl=3600, show_spinner=False)
def fetch_cn_calendar(start, end):
    raw = yf.download("000001.SS", start=start, end=end, auto_adjust=True, progress=False)
    if raw is None or raw.empty:
        raise ValueError("无法获得境内交易日历，不能固定共同日期。")
    close = raw["Close"]
    if isinstance(close, pd.DataFrame): close = close.iloc[:, 0]
    return pd.DatetimeIndex(close.dropna().index).tz_localize(None).normalize()


def fixed_scan_grid(benchmarks, cn_dates, settings):
    # Freeze independently of fund membership. A bad fund cannot move everyone else's window.
    base = benchmarks.dropna().sort_index()
    if settings["lag"]: base = base.shift(settings["lag"]).dropna()
    buffer = int(settings["publication_buffer"])
    if buffer: base = base.iloc[:-buffer]
    n = int(settings["lookback"])
    if len(base) < n + 1: raise ValueError("基准历史不足请求窗口及净值公布缓冲期。")
    requested = base.tail(n + 1)
    dates = requested.index.intersection(cn_dates).sort_values()
    if len(dates) < max(31, int(.7 * len(requested))):
        raise ValueError("境内/美股共同交易日不足，不缩短窗口生成排名。")
    grid = requested.loc[dates]
    info = {"请求开始": str(requested.index[0].date()), "请求结束": str(requested.index[-1].date()),
            "实际开始": str(dates[0].date()), "实际结束": str(dates[-1].date()),
            "共同收益区间数": len(dates) - 1, "固定时滞": settings["lag"]}
    return grid, info


def fetch_scan_nav(code):
    """One bounded, retryable request per fund; reuse connections per worker.
    历史分页接口每页仅20条；本接口一次取回历史，只将评分窗口写入检查点。
    """
    if not re.fullmatch(r"\d{6}", code): raise ValueError("无效基金代码")
    if not hasattr(_nav_local, "session"):
        _nav_local.session = requests.Session()
        _nav_local.session.headers.update(PUBLIC_HEADERS)
    for attempt in range(2):
        try:
            response = _nav_local.session.get(f"https://fund.eastmoney.com/pingzhongdata/{code}.js", timeout=(6, 20))
            response.raise_for_status()
            response.encoding = "utf-8-sig"
            return parse_fund_nav(response.text, code)
        except requests.RequestException:
            if attempt: raise
            time.sleep(1)


def evaluate_scan_fund(code, nav, grid, bonus):
    if nav.empty or nav.index[0] > grid.index[0]:
        raise ValueError("历史不足统一起点")
    if not grid.index.isin(nav.index).all():
        missing = grid.index.difference(nav.index)
        raise ValueError(f"缺少统一采样日期 {len(missing)} 天（首个 {missing[0].date()}），未补造收益")
    window = nav.loc[grid.index[0]:grid.index[-1]]
    if (window["nav"] <= 0).any(): raise ValueError("非正净值")
    wealth = fund_wealth(window, grid.index[0], grid.index[-1]).reindex(grid.index)
    wealth = wealth / wealth.iloc[0]
    rows = [{"基金代码": code, "比较基准": name, **score_pair(wealth, grid[name], bonus)} for name in BENCHMARKS]
    # JSON cannot carry NaN: >=60 returns still needs >=3 complete 20-interval blocks for a bonus.
    rows = json.loads(pd.DataFrame(rows).to_json(orient="records", force_ascii=False, double_precision=12))
    return {"status": "ok", "rows": rows, "path": pack_series(wealth)}


def _scan_one(code, grid, bonus):
    try:
        nav = fetch_scan_nav(code)
    except Exception as exc:
        return {"status": "retry", "reason": f"净值获取/解析失败：{exc}"}
    try:
        return evaluate_scan_fund(code, nav, grid, bonus)
    except ValueError as exc:
        return {"status": "excluded", "reason": str(exc)}


def open_scan_db(job_id):
    if not re.fullmatch(r"[0-9a-f]{24}", job_id): raise ValueError("无效扫描标识")
    FUND_STORE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(FUND_STORE / f"scan_{job_id}.sqlite", timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    db.execute("CREATE TABLE IF NOT EXISTS outcomes (code TEXT PRIMARY KEY, payload TEXT NOT NULL)")
    return db


def scan_lease(owner=None):
    """Cross-session/process lease: only one scan writes shared state at a time."""
    FUND_STORE.mkdir(parents=True, exist_ok=True)
    db = sqlite3.connect(FUND_STORE / "control.sqlite", timeout=30)
    db.execute("CREATE TABLE IF NOT EXISTS lease (id INTEGER PRIMARY KEY, owner TEXT, expiry REAL)")
    db.execute("BEGIN IMMEDIATE")
    row = db.execute("SELECT owner,expiry FROM lease WHERE id=1").fetchone()
    now = time.time()
    if row and row[1] > now and row[0] != owner:
        db.close(); raise ValueError("另一会话正在扫描，请稍后查看保存结果；中断后的占用最多两分钟自动释放。")
    owner = owner or uuid.uuid4().hex
    db.execute("INSERT OR REPLACE INTO lease VALUES (1,?,?)", (owner, now + 120))
    db.commit(); db.close()
    return owner


def release_scan_lease(owner):
    db = sqlite3.connect(FUND_STORE / "control.sqlite")
    try:
        db.execute("DELETE FROM lease WHERE owner=?", (owner,))
        db.commit()
    finally:
        db.close()


def prepare_scan(catalog, settings, mode, end_date, previous=None, nonce=""):
    eligible = active_candidates(catalog)
    pools = (previous or {}).get("pools", {})
    codes = sorted(eligible["基金代码"].tolist()) if mode == "full" else sorted({c for pool in pools.values() for c in pool})
    if not codes: raise ValueError("候选池为空，请先运行全量扫描。")
    # Update must keep the methodology and membership selected by the last full scan.
    params = {k: settings[k] for k in SCAN_DEFAULTS if k not in ("auto", "workers")}
    signature = {"schema": SCAN_SCHEMA, "params": params, "codes": codes,
                 "mode": mode, "end": str(end_date), "pools": pools if mode != "full" else {}, "nonce": nonce}
    job_id = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    db = open_scan_db(job_id)
    stored = db.execute("SELECT value FROM meta WHERE key='job'").fetchone()
    db.close()
    if stored:
        return job_id, json.loads(stored[0])
    end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    start = end - pd.Timedelta(days=int(params["lookback"] * 1.9) + 120)
    benchmarks = fetch_screen_benchmarks(str(start.date()), str(end.date()), params["kind"],
                                       "人民币" if params["currency"] == "人民币" else "美元")
    cn_dates = fetch_cn_calendar(str(start.date()), str(end.date()))
    grid, info = fixed_scan_grid(benchmarks, cn_dates, params)
    meta = {**signature, "info": info, "grid": {n: pack_series(grid[n]) for n in BENCHMARKS},
            "catalog": catalog.loc[catalog["基金代码"].isin(codes)].to_dict("records"),
            "created_at": datetime.now(timezone.utc).isoformat()}
    db = open_scan_db(job_id)
    db.execute("INSERT INTO meta VALUES ('job',?)", (json.dumps(meta, ensure_ascii=False),))
    db.commit(); db.close()
    return job_id, meta


def run_persistent_scan(job_id, workers=6, progress=None):
    owner = scan_lease()
    db = open_scan_db(job_id)
    executor = None
    started = time.perf_counter()
    try:
        row = db.execute("SELECT value FROM meta WHERE key='job'").fetchone()
        if not row: raise ValueError("检查点缺少扫描设置。")
        meta = json.loads(row[0]); codes = meta["codes"]
        state = read_fund_state()
        state["pending"] = job_id
        atomic_json(FUND_STORE / "state.json", state)
        outcomes = {c: json.loads(p) for c,p in db.execute("SELECT code,payload FROM outcomes")}
        pending = [c for c in codes if outcomes.get(c, {}).get("status") not in ("ok", "excluded")]
        done = len(codes) - len(pending); resumed = done; processed = 0
        grid = pd.DataFrame({name: unpack_series(p) for name,p in meta["grid"].items()})
        workers = max(1, min(12, int(workers)))
        executor = ThreadPoolExecutor(max_workers=workers)
        iterator = iter(pending); jobs = {}
        def submit_one():
            code = next(iterator, None)
            if code: jobs[executor.submit(_scan_one, code, grid, meta["params"]["bonus"])] = code
        for _ in range(workers): submit_one()
        if progress: progress(done, len(codes), 0, None)
        while jobs:
            finished, _ = wait(jobs, timeout=1, return_when=FIRST_COMPLETED)
            scan_lease(owner)
            for future in finished:
                code = jobs.pop(future)
                payload = future.result()
                outcomes[code] = payload
                db.execute("INSERT OR REPLACE INTO outcomes VALUES (?,?)", (code, json.dumps(payload, ensure_ascii=False, allow_nan=False)))
                db.commit()  # Each completed fund survives reruns/restarts.
                done += 1; processed += 1
                elapsed = time.perf_counter() - started
                eta = (len(codes) - done) * elapsed / processed
                if progress: progress(done, len(codes), elapsed, eta)
                submit_one()
        rows = []; errors = []; paths = {n: pack_series(grid[n]/grid[n].iloc[0]) for n in BENCHMARKS}
        for code in codes:
            item = outcomes[code]
            if item["status"] == "ok": rows.extend(item["rows"])
            else: errors.append({"基金代码": code, "原因": item["reason"], "状态": item["status"]})
        ranking = pd.DataFrame(rows)
        failed = sum(o["status"] == "retry" for o in outcomes.values())
        if ranking.empty or failed > max(3, .2 * len(codes)):
            raise ValueError(f"有效结果不足或下载失败过多（{failed}/{len(codes)}）；进度已保存，旧榜保留，可继续/重试。")
        if meta["mode"] == "full":
            pools = {name: benchmark_ranking(ranking, name, meta["params"]["pool_min_corr"]).head(50)["基金代码"].tolist() for name in BENCHMARKS}
        else:
            pools = meta["pools"]  # Keep all original members even if temporarily excluded/under threshold.
        union = set(c for pool in pools.values() for c in pool)
        for code in union:
            if outcomes.get(code, {}).get("status") == "ok": paths[code] = outcomes[code]["path"]
        now = datetime.now(timezone.utc).isoformat()
        snapshot = {"schema": SCAN_SCHEMA, "job_id": job_id, "computed_at": now, "mode": meta["mode"],
                    "params": meta["params"], "info": meta["info"], "catalog": meta["catalog"],
                    "ranking": rows, "errors": errors, "paths": paths, "pools": pools,
                    "coverage": {"total": len(codes), "valid": sum(o["status"] == "ok" for o in outcomes.values()),
                                 "excluded": sum(o["status"] == "excluded" for o in outcomes.values()),
                                 "failed": failed, "resumed": resumed, "processed": processed,
                                 "elapsed_seconds": time.perf_counter() - started}}
        atomic_json(FUND_STORE / f"result_{job_id}.json", snapshot)
        state = read_fund_state()
        state.update({"latest": job_id, "pending": None})
        if meta["mode"] == "full": state.update({"full": job_id, "full_at": now})
        atomic_json(FUND_STORE / "state.json", state)
        return snapshot
    finally:
        if executor: executor.shutdown(wait=True, cancel_futures=True)
        db.close()
        release_scan_lease(owner)


def load_fund_snapshot(job_id):
    if not job_id: return None
    if not re.fullmatch(r"[0-9a-f]{24}", job_id): raise ValueError("无效扫描标识")
    return json.loads((FUND_STORE / f"result_{job_id}.json").read_text(encoding="utf-8"))


def fund_backup_bytes(state):
    documents = {"state.json": state}
    for job_id in {state.get("full"), state.get("latest")} - {None}:
        documents[f"result_{job_id}.json"] = load_fund_snapshot(job_id)
    buf = BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, data in documents.items():
            z.writestr(name, json.dumps(data, ensure_ascii=False, allow_nan=False))
    return buf.getvalue()


def restore_fund_backup(content):
    # Only JSON; no pickle, arbitrary extraction paths, or uploaded SQLite execution.
    with zipfile.ZipFile(BytesIO(content)) as z:
        if len(z.infolist()) > 3 or len(set(z.namelist())) != len(z.namelist()) or sum(i.file_size for i in z.infolist()) > 80_000_000:
            raise ValueError("基金备份大小或文件数量不符。")
        state = json.loads(z.read("state.json"))
        identifiers = {state.get("full"), state.get("latest")} - {None}
        if any(not re.fullmatch(r"[0-9a-f]{24}", str(i)) for i in identifiers): raise ValueError("无效快照编号")
        expected = {"state.json"} | {f"result_{i}.json" for i in identifiers}
        if set(z.namelist()) != expected: raise ValueError("基金备份内容不符")
        snapshots = {i: json.loads(z.read(f"result_{i}.json")) for i in identifiers}
    if not state.get("full") or not state.get("latest"): raise ValueError("缺少完整候选池快照")
    for job_id, snap in snapshots.items():
        if snap.get("schema") != SCAN_SCHEMA or snap.get("job_id") != job_id: raise ValueError("基金备份版本或标识不符")
        if set(snap["pools"]) != set(BENCHMARKS): raise ValueError("备份缺少三个独立候选池")
        for pool in snap["pools"].values():
            if len(pool) > 50 or len(set(pool)) != len(pool) or any(not re.fullmatch(r"\d{6}", c) for c in pool):
                raise ValueError("候选代码或数量不符")
        for path in snap["paths"].values(): unpack_series(path)
        frame = pd.DataFrame(snap["ranking"])
        if frame.empty or not {"基金代码", "比较基准", "综合得分", "收益相关系数"}.issubset(frame.columns):
            raise ValueError("备份评分字段不符")
        if not set(frame["比较基准"]).issubset(BENCHMARKS) or frame.duplicated(["基金代码", "比较基准"]).any():
            raise ValueError("备份评分基准或唯一性不符")
    owner = scan_lease()
    try:
        for job_id,snap in snapshots.items(): atomic_json(FUND_STORE / f"result_{job_id}.json", snap)
        state["pending"] = None
        # Imported files never authorize an automatic network job on this server.
        state["auto"] = False
        atomic_json(FUND_STORE / "state.json", state)
    finally: release_scan_lease(owner)


def next_full_due(state):
    if not state.get("full_at"): return True
    return pd.Timestamp.now(tz="UTC") >= pd.Timestamp(state["full_at"]) + pd.DateOffset(months=1)


def render_saved_fund_result(snapshot):
    params = snapshot["params"]; cov = snapshot["coverage"]; info = snapshot["info"]
    saved_time = pd.Timestamp(snapshot["computed_at"]).tz_convert("Asia/Shanghai").strftime("%Y-%m-%d %H:%M:%S") + "（北京时间）"
    st.caption(f"已保存：{saved_time} · {'全量扫描' if snapshot['mode']=='full' else '月度候选池刷新'}。"
               f"处理 {cov['total']:,}，有效 {cov['valid']:,}，数据不符 {cov['excluded']:,}，获取失败 {cov['failed']:,}。")
    st.caption(f"统一比较 {info['实际开始']} 至 {info['实际结束']}，{info['共同收益区间数']} 个收益区间；"
               f"{params['kind']} / {params['currency']} / 时滞 {params['lag']}。保留 {params['publication_buffer']} 个美股交易日作为净值公布缓冲。")
    st.caption("日期由基准与境内交易日历预先固定，缺少其中任意采样日的基金单独排除。"
               "不会因为加入异常基金改变其他基金的评分日期；节假日收益按实际间隔累计。")
    if cov["failed"]: st.warning(f"有 {cov['failed']} 个份额获取失败；当前排名不涵盖这些份额，可继续/重试最近扫描。")
    if snapshot["errors"]:
        with st.expander("未参与评分的基金及原因"):
            errors = pd.DataFrame(snapshot["errors"])
            st.dataframe(errors, hide_index=True, width="stretch")
            st.download_button("下载排除及失败清单", errors.to_csv(index=False).encode("utf-8-sig"), "fund_errors.csv")
    catalog = pd.DataFrame(snapshot["catalog"])
    ranking = pd.DataFrame(snapshot["ranking"]).merge(catalog, on="基金代码", how="left", validate="many_to_one")
    ranking["基金类别"] = np.where(ranking["基金类型"].str.startswith("QDII", na=False), "主动 QDII", "境内股票/混合（非QDII）")
    result = {"paths": {n: unpack_series(p) for n,p in snapshot["paths"].items()}, "info": info,
              "computed_at": snapshot["computed_at"]}
    st.markdown("### 三个独立的月度候选池（每榜最多50个份额）")
    st.caption(f"入池相关系数门槛 {params['pool_min_corr']:.2f}；不足50个合格份额时不凑数。"
               "刷新仅重新排列本榜原有成员，月底全量扫描再纳入新机会。A/C份额分别计分。")
    tabs = st.tabs([f"{n}相似基金" for n in BENCHMARKS])
    for tab,name in zip(tabs, BENCHMARKS):
        with tab:
            board_rows = ranking.loc[(ranking["比较基准"] == name) & ranking["基金代码"].isin(snapshot["pools"][name])]
            if board_rows.empty:
                st.info("本榜暂无满足条件且数据完整的候选。下次全量扫描会重新筛选。")
            else: render_benchmark_board(board_rows, name, result, catalog, params["kind"], params["currency"])
    with st.expander("评分方法及全量结果"):
        st.markdown("各指数独立评分：相似度满分100，其中55%收益相关性、25%累计路径接近度、20%最终收益接近度。"
                    "后两项使用10个百分点衰减尺度；负相关不加分。总超额为正、相关性≥0.6、至少3个完整20区间分段且多数跑赢，"
                    f"才按幅度及持续性加分（上限{params['bonus']:g}）。不合成三个指数。")
        st.caption("基金按公布增长率复利计算，未扣个人申赎费。ETF收益代理包含费用与复权调整；价格指数不含分红。"
                   "人民币基准使用USD/CNY市场汇率；日期为事后对齐，不代表可提前交易。"
                   "境内类别不保证持仓全部为A股，当前存续目录存在存续偏差，历史相似不等于未来收益保证。"
                   "公开候选目录不等于支付宝在售清单，额度需自行确认。")
        export = ranking.copy()
        for key in ["实际开始", "实际结束", "共同收益区间数"]: export[key] = info[key]
        export["口径"] = params["kind"]; export["币种"] = params["currency"]
        export["扫描类型"] = snapshot["mode"]; export["计算时间"] = snapshot["computed_at"]
        st.download_button("下载本次全部评分 CSV", export.to_csv(index=False).encode("utf-8-sig"), "fund_all_scores.csv")
        st.markdown("[天天基金公开净值](https://fund.eastmoney.com/) · [Yahoo Finance](https://finance.yahoo.com/) · "
                    "[SOXQ指数收益代理](https://www.invesco.com/us/en/financial-products/etfs/invesco-phlx-semiconductor-etf.html)")


def render_fund_screener():
    st.subheader("主动基金 · 月度全量扫描与三个独立候选池")
    if st.session_state.pop("fund_scan_saved_notice", False):
        st.success("已保存。下次打开网页直接读取榜单；全量扫描进度也已保留。")
    st.caption("境内主动股票/混合 + 主动QDII，不按名称或持仓地区预选。每月全面发现候选，平时仅刷新三个前50的去重并集（最多150个份额）。")
    try:
        state = read_fund_state()
        snapshot = load_fund_snapshot(state.get("latest"))
    except Exception as exc:
        st.error(f"保存结果读取失败，请从基金榜单ZIP恢复：{exc}"); state = {}; snapshot = None
    with st.expander("保存与恢复"):
        st.caption("扫描检查点与榜单保存在运行服务器的 fund_data 目录。Streamlit Cloud休眠/重建后不保证本地文件保留，"
                   "请下载基金榜单备份；它包含月度候选池、已完成评分及最近结果，不包含未完成扫描的检查点。")
        if snapshot:
            st.download_button("下载基金榜单与候选池备份 ZIP", fund_backup_bytes(state), "fund_screen_backup.zip", "application/zip")
        upload = st.file_uploader("恢复基金榜单备份（与泡沫指数行情ZIP不同）", type=["zip"], key="fund_restore")
        if upload is not None and st.button("恢复该基金榜单备份"):
            try:
                restore_fund_backup(upload.getvalue())
                for key in ("fund_lookback", "fund_bonus", "fund_kind", "fund_currency", "fund_lag",
                            "fund_buffer", "fund_pool_corr", "fund_auto"):
                    st.session_state.pop(key, None)
                st.rerun()
            except Exception as exc: st.error(f"恢复失败：{exc}")
    initial = {**SCAN_DEFAULTS, **(snapshot or {}).get("params", {})}
    with st.expander("全量扫描设置（修改后需重新全量扫描）", expanded=snapshot is None):
        c1,c2,c3 = st.columns(3)
        lookback = int(c1.number_input("回看美股交易日数", 60, 1500, int(initial["lookback"]), key="fund_lookback"))
        bonus = float(c2.number_input("稳定超额最高加分", 0.0, 50.0, float(initial["bonus"]), key="fund_bonus"))
        workers = int(c3.number_input("并发下载数", 1, 12, 6, key="fund_workers"))
        c1,c2,c3 = st.columns(3)
        kinds = ["ETF复权收益代理", "原始价格指数"]
        currencies = ["人民币", "美元指数对人民币基金（未校正）"]
        kind = c1.selectbox("比较口径", kinds, index=kinds.index(initial["kind"]), key="fund_kind")
        currency = c2.selectbox("比较币种", currencies, index=currencies.index(initial["currency"]), key="fund_currency")
        lag = c3.selectbox("基金日期对应美股日期", [0,1], index=int(initial["lag"]), format_func=lambda x: "同一日期（事后比较）" if x==0 else "前一个美股交易日", key="fund_lag")
        c1,c2 = st.columns(2)
        buffer = int(c1.number_input("净值公布缓冲（美股交易日）", 0, 10, int(initial["publication_buffer"]), key="fund_buffer"))
        mincorr = float(c2.number_input("月度候选入池最低相关系数", 0.0, 1.0, float(initial["pool_min_corr"]), step=.05, key="fund_pool_corr"))
        st.caption("并发主要缩短网络等待，不是按CPU核数强行增加连接。默认6路；遇到限流可降低。"
                   "缓冲期用于等待净值公布，榜单展示的是实际净值日期，不是盘中估值。")
    settings = {"lookback": lookback, "bonus": bonus, "workers": workers, "kind": kind, "currency": currency,
                "lag": lag, "publication_buffer": buffer, "pool_min_corr": mincorr, "auto": True}
    params_changed = snapshot is not None and any(settings[k] != snapshot["params"][k] for k in snapshot["params"])
    if params_changed: st.info("设置已变化；下面仍是标明原参数的已保存榜单，候选刷新沿用原口径，新设置需全量扫描后生效。")
    auto = st.checkbox("到期后自动更新：全量每月一次，候选每小时最多刷新一次", value=state.get("auto", True), key="fund_auto")
    st.caption("在打开或操作网页时检查是否到期；网页关闭/服务器休眠时不运行后台定时任务。每小时刷新仅获取最新已公布日净值。")
    if auto != state.get("auto"):
        state["auto"] = auto; atomic_json(FUND_STORE / "state.json", state)
    c1,c2,c3 = st.columns(3)
    full_clicked = c1.button("全量扫描 / 重新选前50", type="primary", key="run_fund_full")
    update_clicked = c2.button("立即刷新候选净值与排名", disabled=snapshot is None, key="run_fund_update")
    resume_id = state.get("pending") or (snapshot or {}).get("job_id")
    can_resume = bool(resume_id and (state.get("pending") or (snapshot or {}).get("coverage", {}).get("failed")) and (FUND_STORE / f"scan_{resume_id}.sqlite").exists())
    resume_clicked = c3.button("继续 / 重试最近扫描", disabled=not can_resume, key="resume_fund_scan")
    if state.get("pending"): st.info("存在未完成的扫描；点击继续会复用已完成记录，不重新下载全部基金。")
    if state.get("full_at"):
        due = pd.Timestamp(state["full_at"]) + pd.DateOffset(months=1)
        st.caption(f"下次全量扫描到期：{due.strftime('%Y-%m-%d')}；候选池刷新不会推迟该日期。")
    automatic = None
    if auto and snapshot and not params_changed and not state.get("pending"):
        age = pd.Timestamp.now(tz="UTC") - pd.Timestamp(snapshot["computed_at"])
        attempted = pd.Timestamp(state.get("last_attempt", "2000-01-01T00:00:00Z"))
        if (pd.Timestamp.now(tz="UTC") - attempted).total_seconds() >= 3600:
            if next_full_due(state): automatic = "full"
            elif age.total_seconds() >= 3600: automatic = "update"
    mode = "full" if full_clicked else "update" if update_clicked else automatic
    progress_slot = st.empty()
    # Paint the saved result before a slow network job, keeping the old board visible.
    if snapshot: render_saved_fund_result(snapshot)
    if mode or resume_clicked:
        try:
            state["last_attempt"] = datetime.now(timezone.utc).isoformat()
            atomic_json(FUND_STORE / "state.json", state)
            if resume_clicked:
                job_id = resume_id
            else:
                with st.spinner("准备目录、统一基准日期和检查点…"):
                    catalog = fetch_fund_catalog() if mode == "full" else pd.DataFrame(load_fund_snapshot(state["full"])["catalog"])
                    use_settings = settings if mode == "full" else {**settings, **snapshot["params"]}
                    previous = load_fund_snapshot(state.get("full"))
                    job_id,_ = prepare_scan(catalog, use_settings, mode, pd.Timestamp.now(tz="Asia/Shanghai").date(), previous,
                                            nonce=state["last_attempt"])
            bar = progress_slot.progress(0.0, text="读取已保存的扫描进度…")
            def progress(done, total, elapsed, eta):
                text = f"已处理 {done:,}/{total:,}；本轮已用 {elapsed/60:.1f} 分钟"
                if eta is not None: text += f"，按当前速度预计剩余 {eta/60:.1f} 分钟"
                bar.progress(done / total, text=text)
            snapshot = run_persistent_scan(job_id, workers, progress)
            st.session_state["fund_scan_saved_notice"] = True
            st.rerun()
        except Exception as exc:
            st.error(f"扫描未完成：{exc}。已保存的旧榜继续可用。")
    if not snapshot: st.info("首次点击全量扫描。扫描会自动遍历整个主动基金目录，不再限制每次300个；完成后显示并保存三个榜单。")



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
    st.set_page_config(page_title="AI泡沫指数 V3.3", page_icon="📈", layout="wide")
    
    # ============================================================
    # Bloomberg / TradingView 深色主题 CSS
    # ============================================================
    st.markdown("""
    <style>
    /* 主背景 */
    .stApp { background-color: #131722; color: #d1d4dc; }
    [data-testid="stHeader"] { background-color: #131722; }
    button[kind="secondary"], [data-testid="stFileUploaderDropzone"],
    [data-testid="stNumberInputContainer"], [data-testid="stNumberInputContainer"] button,
    input, textarea { background-color: #1e222d !important; color: #d1d4dc !important; }
    button[kind="secondary"] { border-color: #4a4e59 !important; }
    button:disabled { opacity: 0.5; }
    
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
