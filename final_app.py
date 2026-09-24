"""AI bubble dashboard V3.1 / 完整模型与早期参考历史分开统计。"""
import json
import hashlib
import os
import zipfile
from pathlib import Path
from datetime import datetime, timezone
from io import BytesIO, StringIO

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import percentileofscore
import requests
import statsmodels.api as sm

# 用户配置 / User configuration: complete-model formula is unchanged.
USER_CONFIG = {
    "START_DATE": "1960-01-01",
    "BIAS_POINTS": 15.0,  # 人工风险修正 / Required manual risk adjustment
    "CACHE_SECONDS": 3600,
    "MISSING_POLICY": "legacy",  # legacy 保持原版前向填充；strict 不补值（会改变部分读数）
    "SAVE_LOCAL_BACKUP": True,
    "BACKUP_DIR": "bubble_data",  # Relative to this script, not the working directory
}
MODEL_VERSION = "3.1-full-and-reference"
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
    st.markdown("# 🛡️ 私人量化终端：AI 泡沫综合指数 V3.1")
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
        st.info("可在侧边栏上传以前下载的 ZIP 行情备份。")
        st.stop()

    stats = percentile_summary(df)
    valid = df[df["总泡沫指数"].notna()]
    if valid.empty:
        st.error("历史长度尚不足以计算任何指数。")
        st.stop()
    full_df = df[df["完整指数"].notna()]
    ref_df = df[df["早期参考指数"].notna()]
    show_early = st.sidebar.checkbox("显示早期参考历史", value=False)
    span = st.sidebar.selectbox("时间轴范围", ["最近 400 个交易日", "最近 1000 个交易日", "全部可用历史"])
    start = valid.index[0] if show_early or full_df.empty else full_df.index[0]
    frame = df.loc[start:].copy()
    if span != "全部可用历史":
        frame = frame.tail(400 if "400" in span else 1000)
    if not show_early:
        frame["早期参考指数"] = np.nan
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

    tabs = st.tabs(["📈 综合指数看板", "🔬 历史回测分析", "🔮 华夏净值预测 (OLS)", "🗂️ 数据与备份"])
    with tabs[0]:
        st.subheader("综合指数走势")
        st.plotly_chart(plot_index(frame), width="stretch")
        if not show_early and len(ref_df):
            st.caption("查看更早曲线：勾选侧边栏“显示早期参考历史”，并选择“全部可用历史”。")
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
        render_ols()
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


@st.cache_data(ttl=3600)
def fetch_ols_data():
    try:
        # 1. 抓取美股两大数据
        end_date = pd.Timestamp.today()
        # 将动态的 60 天回溯，改为精准锚定今年的 4 月 29 日
        current_year = end_date.year
        start_date = pd.Timestamp(f'{current_year}-04-29')
        
        us_data = yf.download(['^NDX', '^SOX'], start=start_date, end=end_date)['Close']
        us_pct = us_data.pct_change(fill_method=None).dropna() * 100
        us_pct = us_pct.rename(columns={'^NDX': 'NDX', '^SOX': 'SOX'})
        if us_pct.index.tz is not None: us_pct.index = us_pct.index.tz_localize(None)

        # 2. 抓取天天基金网数据
        url = "http://api.fund.eastmoney.com/f10/lsjz?fundCode=005698&pageIndex=1&pageSize=60"
        headers = {"Referer": "http://fundf10.eastmoney.com/"}
        res = requests.get(url, headers=headers, timeout=5).json()
        fund_df = pd.DataFrame(res['Data']['LSJZList'])
        fund_df['FSRQ'] = pd.to_datetime(fund_df['FSRQ'])
        fund_df['Fund'] = pd.to_numeric(fund_df['JZZZL'], errors='coerce')
        fund_pct = fund_df.set_index('FSRQ')['Fund'].sort_index()

        # 3. 智能假期对齐与复利合并
        df_combined = us_pct.copy()
        df_combined['Fund_Raw'] = fund_pct
        
        # 使用 bfill 将美股交易日映射到下一个最近的基金净值更新日
        df_combined['Period_End'] = df_combined['Fund_Raw'].notna().replace(False, np.nan)
        df_combined['Period_End'] = df_combined.index.where(df_combined['Period_End'].notna())
        df_combined['Period_End'] = df_combined['Period_End'].bfill()

        aligned_data = []
        for period_end, group in df_combined.groupby('Period_End'):
            if pd.isna(period_end): continue
            
            # 累乘计算期间的美股复利收益率
            cum_ndx = ((1 + group['NDX'] / 100).prod() - 1) * 100
            cum_sox = ((1 + group['SOX'] / 100).prod() - 1) * 100
            fund_ret = group['Fund_Raw'].iloc[-1]
            
            # 格式化日期标签
            start_dt = group.index[0].strftime('%m-%d')
            end_dt = group.index[-1].strftime('%m-%d')
            date_label = f"{start_dt} 至 {end_dt}" if start_dt != end_dt else start_dt
            
            aligned_data.append({
                '交易区间': date_label,
                'Date': period_end,
                'NDX': cum_ndx,
                'SOX': cum_sox,
                'Fund': fund_ret
            })

        final_df = pd.DataFrame(aligned_data).set_index('Date')
        final_df['是否纳入回归'] = True 
        
        # 取消尾部截断，使用日期严格过滤，保留 4.29 至今的所有有效拟合样本
        final_df = final_df[final_df.index >= start_date]
        
        return final_df
    except Exception as e:
        return pd.DataFrame()

# ============================================================
# 新增模块：自动抓取最新单日涨跌幅 (用于净值模拟器默认值)
# ============================================================
@st.cache_data(ttl=1800)
def get_latest_market_returns():
    try:
        # 抓取最近 5 天数据以确保至少有两个有效交易日来计算涨跌幅
        df_recent = yf.download(['^NDX', '^SOX'], period='5d')['Close']
        pct_recent = df_recent.pct_change(fill_method=None).dropna() * 100
        dt_str = pct_recent.index[-1].strftime('%Y-%m-%d')
        val_ndx = float(pct_recent['^NDX'].iloc[-1])
        val_sox = float(pct_recent['^SOX'].iloc[-1])
        return dt_str, val_ndx, val_sox
    except Exception:
        return "未知日期", 0.0, 0.0



def render_ols():
    st.caption("OLS 为独立模块，不参与泡沫指数或两种百分位。当前基金接口最多取最近 60 条净值记录。")
    st.subheader("🔮 华夏全球科技先锋 (005698) 双因子 OLS 预测")
    st.markdown("""
    <p style='color:#787b86;font-size:1.0rem;'>
    本模块自动抓取近期纳斯达克(NDX)、半导体(SOX)及该基金的实际每日涨跌幅，通过多元线性回归估计基金对两个指数的统计敏感度（不等同于实际持仓）。
    </p>
    """, unsafe_allow_html=True)
    
    valid_data = pd.DataFrame()
    model = None
    ols_data = fetch_ols_data()
    
    if ols_data.empty:
        st.error("获取 OLS 回归数据失败，请检查网络。")
    else:
        col_table, col_model = st.columns([1.2, 1])
        
        with col_table:
            st.markdown("**1. 数据清洗与校准** (按当前日期合并规则处理；取消勾选可排除样本)")
            
            # 调整显示顺序，加入区间标签
            display_cols = ['交易区间', 'NDX', 'SOX', 'Fund', '是否纳入回归']
            edited_df = st.data_editor(
                ols_data[display_cols].style.format("{:.2f}", subset=['NDX', 'SOX', 'Fund'], na_rep="空"),
                column_config={
                    "交易区间": st.column_config.TextColumn("交易区间"),
                    "是否纳入回归": st.column_config.CheckboxColumn("参与回归?", default=True)
                },
                width="stretch", height=300
            )
        
        # 运行回归
        valid_data = edited_df[edited_df['是否纳入回归'] == True].dropna(subset=['NDX', 'SOX', 'Fund'])
        
        with col_model:
            if len(valid_data) < 5:
                st.warning("⚠️ 请至少保留 5 天的有效数据以运行回归模型。")
            else:
                X = valid_data[['NDX', 'SOX']]
                X = sm.add_constant(X, has_constant='add')
                y = valid_data['Fund']
                
                model = sm.OLS(y, X).fit()
                alpha = model.params['const']
                beta_ndx = model.params.get('NDX', 0)
                beta_sox = model.params.get('SOX', 0)
                
                st.markdown("**2. 模型估计的因子敏感度**")
                st.info(f"**方程：** 基金收益 = {alpha:.2f}% + ({beta_ndx:.2f} × NDX) + ({beta_sox:.2f} × SOX)")
            
                c1, c2, c3 = st.columns(3)
                c1.metric("纳指敞口 (Beta)", f"{beta_ndx:.2f}")
                c2.metric("半导体敞口 (Beta)", f"{beta_sox:.2f}")
                c3.metric("拟合度 (R²)", f"{model.rsquared:.2f}")
            
    st.markdown("---")
    st.markdown("### 🎯 净值模拟器")
    
    latest_dt, auto_ndx, auto_sox = get_latest_market_returns()
    st.markdown(f"<p style='color:#787b86;font-size:0.95rem;'>🤖 已自动同步美股最新交易日 (<b>{latest_dt}</b>) 的真实收盘数据。你也可以在下方手动修改进行沙盘推演：</p>", unsafe_allow_html=True)
    
    pred_col1, pred_col2, pred_col3 = st.columns([1, 1, 1.5])
    with pred_col1:
        in_ndx = st.number_input("👉 当日 NDX 涨跌幅 (%)", value=round(auto_ndx, 2), step=0.1)
    with pred_col2:
        in_sox = st.number_input("👉 当日 SOX 涨跌幅 (%)", value=round(auto_sox, 2), step=0.1)
        
    if model is not None and len(valid_data) >= 5:
        pred_val = alpha + beta_ndx * in_ndx + beta_sox * in_sox
        pred_x = pd.DataFrame({'const': [1.0], 'NDX': [in_ndx], 'SOX': [in_sox]})
        interval = model.get_prediction(pred_x[model.model.exog_names]).summary_frame(alpha=.05).iloc[0]
        pred_low, pred_high = float(interval['obs_ci_lower']), float(interval['obs_ci_upper']) 
        
        with pred_col3:
            st.markdown("**95% 预测区间结果：**")
            st.metric(
                "核心预测值", 
                f"{pred_val:.2f}%", 
                f"波动范围: [{pred_low:.2f}%, {pred_high:.2f}%]", 
                delta_color="off"
            )


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
    st.set_page_config(page_title="AI泡沫指数 V3.1", page_icon="📈", layout="wide")
    
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
