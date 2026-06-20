import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.stats import percentileofscore
from io import StringIO
import requests
import statsmodels.api as sm

# ============================================================
# 全局配置
# ============================================================
st.set_page_config(page_title="AI泡沫指数终极看板", page_icon="📈", layout="wide")

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
C_BG       = "#131722"
C_PANEL    = "#1e222d"
C_BORDER   = "#2a2e39"
C_BLUE     = "#2962ff"
C_TEXT     = "#d1d4dc"
C_MUTED    = "#787b86"

ZONES = [
    (0,  20,  "💎 史诗底部 (0-20)",   "#34C759"),
    (20, 30,  "🟩 恐慌底部 (20-30)",  "#30D158"),
    (30, 40,  "🟢 偏低区域 (30-40)",  "#90EE90"),
    (40, 70,  "🟡 中性震荡 (40-70)",  "#FFD700"),
    (70, 80,  "🟠 偏高区域 (70-80)",  "#FF9F0A"),
    (80, 90,  "🔴 高风险区 (80-90)",  "#FF6B35"),
    (90, 101, "🚨 极度危险 (90+)",    "#FF3B30"),
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

# ============================================================
# 数据获取 & 计算（完全还原你的原始手工调校逻辑）
# ============================================================
@st.cache_data(ttl=3600)
def fetch_and_calculate():
    # 1. 下载底层数据
    tickers = ["QQQ", "^VIX", "SPHB", "SPLV", "IPO", "SPY", "HYG", "IEF", "^TNX"]
    raw = yf.download(tickers, start="2012-01-01")
    close = raw['Close'].ffill()
    volume = raw['Volume'].ffill()

    def get_pct(series, window):
        return series.rolling(window).apply(
            lambda x: pd.Series(x).rank(pct=True).iloc[-1] * 100 if len(x) >= window / 2 else np.nan)

    # ==========================================
    # 模块一：情绪指标 (P1 - P4)
    # ==========================================
    sma200 = close['QQQ'].rolling(200).mean()
    p1 = get_pct((close['QQQ'] - sma200) / sma200, 2520)

    p2 = get_pct(1 / close['^VIX'], 2520)

    p3_raw = get_pct(close['SPHB'] / close['SPLV'], 756)
    p3 = 50 + (p3_raw - 50) * 0.4  # 恢复原版压缩系数

    p4_enhanced = (close['IPO'] / close['SPY']) * (volume['IPO'] / volume['IPO'].rolling(126).mean())
    p4 = get_pct(p4_enhanced, 756)

    # 情绪合成 (保留调校魔法)
    sentiment_raw = p1 * 0.3 + p2 * 0.3 + p3 * 0.1 + p4 * 0.3
    sentiment_smoothed = sentiment_raw.rolling(10).mean()
    sentiment_index = 20 + (sentiment_smoothed - 20) * 0.83  # 恢复原版压缩系数

    # ==========================================
    # 模块二：资金指标 (P5 - P6)
    # ==========================================
    # P5 流动性 (高低利差平替)
    p5_raw = get_pct(close['HYG'] / close['IEF'], 756).rolling(10).mean()
    p5_final = (80 - (100 - p5_raw) * 3.0).clip(lower=0, upper=100)

    # P6 降息预期 (美债收益率动量阶梯化)
    tnx_change = close['^TNX'] - close['^TNX'].shift(20)
    smoothed_change = tnx_change.rolling(10).mean()

    def step_fn(c):
        if pd.isna(c): return np.nan
        if c < -0.25: return 100
        elif c < -0.05: return 75
        elif c < 0.15: return 50
        else: return 25

    p6_final = smoothed_change.apply(step_fn).ffill()

    # 资金合成 (等权结合 P5 和 P6)
    capital_index = (p5_final + p6_final) / 2

    # ==========================================
    # 🚀 终极总合成：情绪 vs 资金 = 2 : 1
    # ==========================================
    total_index = (sentiment_index * 2 + capital_index * 1) / 3
    total_smoothed = total_index.rolling(10).mean()
    
    # 🚀 最终修正魔法：完全还原你的 +15 曲线上移逻辑，拒绝盲目归一化导致局部的低点变成绝对恐慌。
    total_smoothed = (total_smoothed + 15).clip(lower=0, upper=100)

    df = pd.DataFrame({
        '总泡沫指数': total_smoothed,
        '综合情绪指标': sentiment_index,
        '综合资金指标': capital_index,
        'QQQ': close['QQQ'],  # 保留 QQQ，供回测模块使用
    }).dropna()

    # 清理时区并修改索引名称为中文
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = '日期'

    return df

# ============================================================
# 回测核心逻辑
# ============================================================
@st.cache_data(ttl=3600)
def run_backtest(df_json: str) -> list[dict]:
    """
    接收 JSON 字符串以规避 Streamlit 无法哈希 DataFrame 的问题。
    返回各区间各周期的统计结果列表。
    """
    df = pd.read_json(StringIO(df_json))
    df.index = pd.to_datetime(df.index, unit='ms')
    df.index.name = '日期'

    qqq = df['QQQ']
    bubble = df['总泡沫指数']
    results = []

    for lo, hi, label, color in ZONES:
        mask = (bubble >= lo) & (bubble < hi)
        dates = df.index[mask]
        row = {'区间': label, '信号天数': int(mask.sum()), 'color': color}

        for period_name, days in PERIODS:
            rets = []
            for d in dates:
                try:
                    loc = qqq.index.get_loc(d)
                    if loc + days < len(qqq):
                        r = (qqq.iloc[loc + days] / qqq.iloc[loc] - 1) * 100
                        rets.append(r)
                except Exception:
                    pass

            if rets:
                row[f'{period_name}_avg']    = round(float(np.mean(rets)), 2)
                row[f'{period_name}_median'] = round(float(np.median(rets)), 2)
                row[f'{period_name}_win']    = round(sum(r > 0 for r in rets) / len(rets) * 100, 1)
                row[f'{period_name}_n']      = len(rets)
            else:
                row[f'{period_name}_avg']    = np.nan
                row[f'{period_name}_median'] = np.nan
                row[f'{period_name}_win']    = np.nan
                row[f'{period_name}_n']      = 0

        results.append(row)
    return results

# ============================================================
# 新增模块：华夏基金 OLS 预测逻辑
# ============================================================
@st.cache_data(ttl=3600)
def fetch_ols_data():
    try:
        # 1. 抓取美股两大数据
        end_date = pd.Timestamp.today()
        # 将动态的 60 天回溯，改为精准锚定今年的 4 月 29 日
        current_year = end_date.year
        start_date = pd.Timestamp(f'{current_year}-04-29')
        
        us_data = yf.download(['^NDX', '^SOX'], start=start_date, end=end_date)['Close']
        us_pct = us_data.pct_change().dropna() * 100
        us_pct.columns = ['NDX', 'SOX']
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
        pct_recent = df_recent.pct_change().dropna() * 100
        dt_str = pct_recent.index[-1].strftime('%Y-%m-%d')
        val_ndx = float(pct_recent['^NDX'].iloc[-1])
        val_sox = float(pct_recent['^SOX'].iloc[-1])
        return dt_str, val_ndx, val_sox
    except Exception:
        return "未知日期", 0.0, 0.0

# ============================================================
# 页面渲染
# ============================================================
st.markdown(
    "<h1>🛡️ 私人量化终端：美投 AI 泡沫综合指数 V3.0 <span style='float: right; font-size: 0.9rem; color: #787b86; padding-top: 15px; font-weight: normal; font-family: \"Courier New\", monospace;'>Built by <b style='color: #2962ff;'>高章磊</b></span></h1>", 
    unsafe_allow_html=True
)

with st.spinner("📡 正在从华尔街同步底层数据..."):
    df = fetch_and_calculate()

if df.empty:
    st.cache_data.clear()
    st.warning("⚠️ 云端网络拥堵或雅虎财经 API 临时限流，未获取到完整数据。缓存已自动清理，请几秒钟后刷新网页重试。")
    st.stop()

# 侧边栏
st.sidebar.header("⚙️ 看板控制台")
days = st.sidebar.slider("时间轴范围 (天)", 100, 1500, 400)
plot_df = df.tail(days)

# 当前状态
val   = float(plot_df['总泡沫指数'].iloc[-1])
delta = val - float(plot_df['总泡沫指数'].iloc[-2]) if len(plot_df) > 1 else 0.0

if   val >= 90: status, emoji = "极度危险 (高度警戒/清仓)", "🚨"
elif val >= 80: status, emoji = "高风险区 (建议逐步减仓)", "🔴"
elif val >= 70: status, emoji = "偏高区域 (暂停定投/观望)", "🟠"
elif val >= 40: status, emoji = "中性震荡 (持有/观望状态)", "🟡"
elif val >= 30: status, emoji = "偏低区域 (推荐开启定投)", "🟢"
elif val >= 20: status, emoji = "恐慌底部 (建议加大定投)", "🟩"
else:           status, emoji = "史诗级大底 (黄金坑/梭哈)", "💎"

# 历史百分位（基于全量历史，不受滑块影响）
all_vals = df['总泡沫指数'].dropna().values
pct_rank = percentileofscore(all_vals, val, kind='rank')
history_start = df.index[0].strftime('%Y-%m')

# 百分位对应提示
if pct_rank >= 90:   pct_note = "历史极高位 ⚠️"
elif pct_rank >= 75: pct_note = "历史偏高"
elif pct_rank >= 25: pct_note = "历史中性"
elif pct_rank >= 10: pct_note = "历史偏低"
else:                pct_note = "历史极低位 💎"

# 顶部指标卡（4列）
col1, col2, col3, col4 = st.columns([1, 2, 1, 1])
col1.metric("🚨 AI 泡沫指数",  f"{val:.1f}", f"{delta:+.2f}", delta_color="inverse")
col2.metric("📊 市场状态评级", f"{emoji} {status}")
col3.metric("📅 最新更新日期", plot_df.index[-1].strftime('%Y-%m-%d'))
col4.metric(f"📐 历史百分位 (since {history_start})", f"{pct_rank:.1f}%", pct_note)

st.markdown("---")

# ============================================================
# Tabs
# ============================================================
tab1, tab2, tab3 = st.tabs(["  📈  综合指数看板  ", "  🔬  历史回测分析  ", "  🔮  华夏净值预测 (OLS)  "])

# ──────────────────────────────────────────────
# Tab 1：综合指数看板
# ──────────────────────────────────────────────
with tab1:
    st.subheader("🌐 综合指数走势 (战术网格全景)")

    # 引入双Y轴
    fig_main = make_subplots(specs=[[{"secondary_y": True}]])

    # 背景色块 (完美对齐网格：使用 x domain 约束宽度，避免溢出到次坐标轴区域)
    for y0, y1, fc, op in [
        (90, 100, "#FF0000", 0.12), (80, 90, "#FF4500", 0.08), (70, 80, "#FFA500", 0.06),
        (30, 40,  "#90EE90", 0.06), (20, 30, "#32CD32", 0.10), (0,  20, "#006400", 0.15),
    ]:
        fig_main.add_shape(
            type="rect",
            xref="x domain", x0=0, x1=1,
            yref="y", y0=y0, y1=y1,
            fillcolor=fc, opacity=op, layer="below", line_width=0
        )

    # 水位网格线
    for y, label, color, dash in [
        (90, "清仓线 (90)",  "#FF3B30", "solid"),
        (80, "减仓线 (80)",  "#FF6B35", "dash"),
        (70, "停投线 (70)",  "#FF9F0A", "dot"),
        (50, "中轴线 (50)",  C_MUTED,   "dash"),
        (40, "开启定投 (40)","#30D158", "dot"),
        (30, "加大定投 (30)","#34C759", "dash"),
        (20, "梭哈底线 (20)","#30D158", "solid"),
    ]:
        pos = "top left" if y >= 50 else "bottom left"
        fig_main.add_hline(
            y=y, line_dash=dash, line_color=color, line_width=1.2,
            annotation_text=label, annotation_position=pos,
            annotation_font_color=color, annotation_font_size=13,
            secondary_y=False
        )

    # 纳指100 (QQQ) 作为背景山脉图 (次坐标轴)
    fig_main.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['QQQ'],
        mode='lines', name='纳指100 (QQQ)',
        line=dict(color='rgba(209, 212, 220, 0.25)', width=1.5), # 半透明的灰白线
        fill='tozeroy', fillcolor='rgba(209, 212, 220, 0.05)',   # 极淡的背景填充
        hovertemplate='纳指QQQ: $%{y:.2f}<extra></extra>',
    ), secondary_y=True)

    # 主曲线 (主坐标轴)
    fig_main.add_trace(go.Scatter(
        x=plot_df.index, y=plot_df['总泡沫指数'],
        mode='lines', name='泡沫指数',
        line=dict(color=C_BLUE, width=2.5),
        hovertemplate='日期: %{x|%Y-%m-%d}<br>泡沫指数: %{y:.2f}<extra></extra>',
    ), secondary_y=False)

    # 更新布局
    layout = dark_layout(height=530)
    fig_main.update_layout(**layout)
    
    # 独立设置双Y轴的范围和网格显示，防止次坐标轴的网格线干扰画面
    fig_main.update_yaxes(title_text="泡沫指数 (0-100)", range=[0, 100], secondary_y=False, title_font=dict(size=15), tickfont=dict(size=14))
    fig_main.update_yaxes(title_text="纳指 QQQ 价格", showgrid=False, secondary_y=True, tickfont=dict(color='rgba(209, 212, 220, 0.5)', size=14), title_font=dict(size=15))

    st.plotly_chart(fig_main, use_container_width=True)

    # 历史分布图
    st.markdown("---")
    st.subheader("📐 历史百分位分布")

    hist_vals = df['总泡沫指数'].dropna().values

    fig_dist = go.Figure()

    # 背景色块（和主图一致）
    for y0, y1, fc, op in [
        (90, 100, "#FF0000", 0.10), (80, 90, "#FF4500", 0.07), (70, 80, "#FFA500", 0.05),
        (30, 40, "#90EE90", 0.05),  (20, 30, "#32CD32", 0.08), (0, 20, "#006400", 0.12),
    ]:
        fig_dist.add_vrect(x0=y0, x1=y1, line_width=0, fillcolor=fc, opacity=op)

    # 历史分布直方图
    fig_dist.add_trace(go.Histogram(
        x=hist_vals, nbinsx=50,
        name='历史分布',
        marker_color='rgba(41,98,255,0.5)',
        marker_line=dict(color='rgba(41,98,255,0.8)', width=0.5),
        hovertemplate='区间: %{x:.1f}<br>天数: %{y}<extra></extra>',
    ))

    # 当前值竖线
    fig_dist.add_vline(
        x=val, line_dash="solid", line_color="#F7DC6F", line_width=2.5,
        annotation_text=f"当前 {val:.1f}  ({pct_rank:.1f}% 百分位)",
        annotation_position="top right",
        annotation_font_color="#F7DC6F",
        annotation_font_size=15,
    )

    layout_dist = dark_layout(height=300, y_title="出现天数")
    layout_dist['xaxis']['title'] = dict(text="泡沫指数值", font=dict(color=C_MUTED, size=15))
    layout_dist['xaxis']['range'] = [0, 100]
    layout_dist['hovermode'] = "x"
    fig_dist.update_layout(**layout_dist)
    st.plotly_chart(fig_dist, use_container_width=True)

    st.markdown(
        f"<p style='color:#787b86;font-size:0.95rem;'>"
        f"基于 {history_start} 至今共 <b style='color:#d1d4dc'>{len(hist_vals)}</b> 个交易日的历史数据。"
        f"当前读数 <b style='color:#F7DC6F'>{val:.1f}</b> 高于历史上 "
        f"<b style='color:#F7DC6F'>{pct_rank:.1f}%</b> 的交易日。</p>",
        unsafe_allow_html=True,
    )
    col_A, col_B = st.columns(2)

    with col_A:
        st.markdown("**🧠 综合情绪指标 (占比 66.7%)**")
        fig_s = go.Figure()
        fig_s.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['综合情绪指标'],
            mode='lines', line=dict(color=C_BLUE, width=2),
            fill='tozeroy', fillcolor='rgba(41,98,255,0.08)',
            hovertemplate='日期: %{x|%Y-%m-%d}<br>情绪: %{y:.2f}<extra></extra>',
        ))
        fig_s.update_layout(**dark_layout(height=260))
        st.plotly_chart(fig_s, use_container_width=True)

    with col_B:
        st.markdown("**💰 综合资金指标 (占比 33.3%)**")
        fig_c = go.Figure()
        fig_c.add_trace(go.Scatter(
            x=plot_df.index, y=plot_df['综合资金指标'],
            mode='lines', line=dict(color='#FF9F0A', width=2),
            fill='tozeroy', fillcolor='rgba(255,159,10,0.08)',
            hovertemplate='日期: %{x|%Y-%m-%d}<br>资金: %{y:.2f}<extra></extra>',
        ))
        fig_c.update_layout(**dark_layout(height=260))
        st.plotly_chart(fig_c, use_container_width=True)


# ──────────────────────────────────────────────
# Tab 2：历史回测分析
# ──────────────────────────────────────────────
with tab2:
    st.subheader("🔬 历史回测：各泡沫区间买入 QQQ 的历史表现")
    st.markdown(
        "<p style='color:#787b86;font-size:0.95rem;'>"
        "统计自 2012 年以来，当泡沫指数处于各区间时买入 QQQ，持有不同周期后的收益情况。"
        "仅供参考，不构成投资建议。</p>",
        unsafe_allow_html=True,
    )

    with st.spinner("⚙️ 正在运行历史回测..."):
        results = run_backtest(df.to_json())

    # ── 汇总表格 ──
    st.markdown("#### 📋 各区间历史收益汇总表")
    table_rows = []
    for r in results:
        row = {'泡沫区间': r['区间'], '历史信号天数': r['信号天数']}
        for p, _ in PERIODS:
            avg = r.get(f'{p}_avg', np.nan)
            win = r.get(f'{p}_win', np.nan)
            row[f'{p} 均收益'] = f"{avg:+.1f}%" if not np.isnan(avg) else "N/A"
            row[f'{p} 胜率']   = f"{win:.0f}%"  if not np.isnan(win) else "N/A"
        table_rows.append(row)

    st.dataframe(pd.DataFrame(table_rows), use_container_width=True, hide_index=True)

    st.markdown("---")

    # ── 选择持有周期 ──
    selected = st.selectbox("📅 选择持有周期查看详细图表", [p[0] for p in PERIODS], index=2)

    avgs   = [r.get(f'{selected}_avg',     np.nan) for r in results]
    meds   = [r.get(f'{selected}_median', np.nan) for r in results]
    wins   = [r.get(f'{selected}_win',     np.nan) for r in results]
    labels = [r['区间'] for r in results]

    # ── 均值 vs 中位数收益 ──
    st.markdown(f"#### 📊 持有 {selected} 的平均 & 中位数收益率 (QQQ)")
    fig_bt = go.Figure()

    fig_bt.add_trace(go.Bar(
        name='平均收益', x=labels, y=avgs,
        marker_color=['rgba(52,199,89,0.85)' if (v or 0) >= 0 else 'rgba(255,59,48,0.85)' for v in avgs],
        # 标签放柱子内部，避免和菱形重叠
        text=[f"{v:+.1f}%" if not np.isnan(v) else "" for v in avgs],
        textposition='inside',
        textfont=dict(color='white', size=13, family='Courier New, monospace'),
        insidetextanchor='middle',
        hovertemplate='%{x}<br>平均收益: %{y:+.1f}%<extra></extra>',
    ))
    fig_bt.add_trace(go.Scatter(
        name='中位数收益', x=labels, y=meds,
        mode='markers',  # 去掉 text mode，数值仅在 hover 显示
        marker=dict(color='#F7DC6F', size=12, symbol='diamond',
                    line=dict(color='white', width=1)),
        hovertemplate='%{x}<br>中位数收益: %{y:+.1f}%<extra></extra>',
    ))
    fig_bt.add_hline(y=0, line_dash="solid", line_color=C_MUTED, line_width=1)

    layout_bt = dark_layout(height=420, y_title="收益率 (%)")
    layout_bt['showlegend'] = True
    layout_bt['legend'] = dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(color=C_TEXT), bgcolor="rgba(0,0,0,0)",
    )
    fig_bt.update_layout(**layout_bt)
    st.plotly_chart(fig_bt, use_container_width=True)

    # ── 胜率图 ──
    st.markdown(f"#### 🎯 持有 {selected} 的正收益胜率 (%)")
    fig_win = go.Figure()

    bar_opacities = [max(0.3, (w or 0) / 100) for w in wins]
    fig_win.add_trace(go.Bar(
        x=labels, y=wins,
        marker_color=[f'rgba(41,98,255,{op:.2f})' for op in bar_opacities],
        text=[f"{w:.0f}%" if not np.isnan(w) else "" for w in wins],
        textposition='inside',
        textfont=dict(color='white', size=13, family='Courier New, monospace'),
        insidetextanchor='middle',
        hovertemplate='%{x}<br>胜率: %{y:.0f}%<extra></extra>',
    ))
    fig_win.add_hline(
        y=50, line_dash="dash", line_color="#FF9F0A", line_width=1.5,
        annotation_text="50% 基准线", annotation_position="top right",
        annotation_font_color="#FF9F0A",
    )

    layout_win = dark_layout(height=360, y_range=[0, 110], y_title="胜率 (%)")
    fig_win.update_layout(**layout_win)
    st.plotly_chart(fig_win, use_container_width=True)

    # ── 四周期折线对比 ──
    st.markdown("---")
    st.markdown("#### 📈 不同持有周期的均值收益率对比（各区间）")
    fig_multi = go.Figure()
    period_colors = [C_BLUE, "#34C759", "#FF9F0A", "#FF3B30"]

    for (pname, _), pcolor in zip(PERIODS, period_colors):
        y_vals = [r.get(f'{pname}_avg', np.nan) for r in results]
        fig_multi.add_trace(go.Scatter(
            name=pname, x=labels, y=y_vals,
            mode='lines+markers',
            line=dict(color=pcolor, width=2),
            marker=dict(size=8, color=pcolor),
            hovertemplate=f'{pname}: %{{y:+.1f}}%<extra></extra>',
        ))

    fig_multi.add_hline(y=0, line_dash="solid", line_color=C_MUTED, line_width=1)

    layout_multi = dark_layout(height=400, y_title="平均收益率 (%)")
    layout_multi['showlegend'] = True
    layout_multi['legend'] = dict(
        orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
        font=dict(color=C_TEXT), bgcolor="rgba(0,0,0,0)",
    )
    fig_multi.update_layout(**layout_multi)
    st.plotly_chart(fig_multi, use_container_width=True)

    st.markdown(
        "<p style='color:#787b86;font-size:0.95rem;text-align:center;'>"
        "⚠️ 历史收益不代表未来表现。本看板为个人研究工具，不构成任何投资建议。</p>",
        unsafe_allow_html=True,
    )

# ──────────────────────────────────────────────
# Tab 3: 华夏基金 OLS 预测核心 (全新增加)
# ──────────────────────────────────────────────
with tab3:
    st.subheader("🔮 华夏全球科技先锋 (005698) 双因子 OLS 预测")
    st.markdown("""
    <p style='color:#787b86;font-size:1.0rem;'>
    本模块自动抓取近期纳斯达克(NDX)、半导体(SOX)及该基金的实际每日涨跌幅，通过多元线性回归测算基金经理真实的底仓暴露度。
    </p>
    """, unsafe_allow_html=True)
    
    ols_data = fetch_ols_data()
    
    if ols_data.empty:
        st.error("获取 OLS 回归数据失败，请检查网络。")
    else:
        col_table, col_model = st.columns([1.2, 1])
        
        with col_table:
            st.markdown("**1. 数据清洗与校准** (已自动合并五一等假期错位，取消勾选可剔除极端值)")
            
            # 调整显示顺序，加入区间标签
            display_cols = ['交易区间', 'NDX', 'SOX', 'Fund', '是否纳入回归']
            edited_df = st.data_editor(
                ols_data[display_cols].style.format("{:.2f}", subset=['NDX', 'SOX', 'Fund'], na_rep="空"),
                column_config={
                    "交易区间": st.column_config.TextColumn("交易区间"),
                    "是否纳入回归": st.column_config.CheckboxColumn("参与回归?", default=True)
                },
                use_container_width=True, height=300
            )
        
        # 运行回归
        valid_data = edited_df[edited_df['是否纳入回归'] == True].dropna(subset=['NDX', 'SOX', 'Fund'])
        
        with col_model:
            if len(valid_data) < 5:
                st.warning("⚠️ 请至少保留 5 天的有效数据以运行回归模型。")
            else:
                X = valid_data[['NDX', 'SOX']]
                X = sm.add_constant(X)
                y = valid_data['Fund']
                
                model = sm.OLS(y, X).fit()
                alpha = model.params['const']
                beta_ndx = model.params.get('NDX', 0)
                beta_sox = model.params.get('SOX', 0)
                
                st.markdown("**2. 模型解析出来的真实底仓**")
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
        
    if len(valid_data) >= 5:
        pred_val = alpha + beta_ndx * in_ndx + beta_sox * in_sox
        pred_se = np.sqrt(model.mse_resid) 
        
        with pred_col3:
            st.markdown("**95% 置信区间预测结果：**")
            st.metric(
                "核心预测值", 
                f"{pred_val:.2f}%", 
                f"波动范围: [{pred_val - 1.96*pred_se:.2f}%, {pred_val + 1.96*pred_se:.2f}%]", 
                delta_color="off"
            )
