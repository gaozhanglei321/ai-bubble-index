import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go

# 网页全局配置
st.set_page_config(page_title="AI泡沫指数终极看板", page_icon="📈", layout="wide")

st.title("🛡️ 私人量化终端：美投 AI 泡沫综合指数 (V2.0 全交互最终版)")
st.markdown("---")


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
    p3 = 50 + (p3_raw - 50) * 0.4

    p4_enhanced = (close['IPO'] / close['SPY']) * (volume['IPO'] / volume['IPO'].rolling(126).mean())
    p4 = get_pct(p4_enhanced, 756)

    # 情绪合成 (保留调校魔法)
    sentiment_raw = p1 * 0.3 + p2 * 0.3 + p3 * 0.1 + p4 * 0.3
    sentiment_smoothed = sentiment_raw.rolling(10).mean()
    sentiment_index = 20 + (sentiment_smoothed - 20) * 0.83

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
        if c < -0.25:
            return 100
        elif c < -0.05:
            return 75
        elif c < 0.15:
            return 50
        else:
            return 25

    p6_final = smoothed_change.apply(step_fn).ffill()

    # 资金合成 (等权结合 P5 和 P6)
    capital_index = (p5_final + p6_final) / 2

    # ==========================================
    # 🚀 终极总合成：情绪 vs 资金 = 2 : 1
    # ==========================================
    total_index = (sentiment_index * 2 + capital_index * 1) / 3
    total_smoothed = total_index.rolling(10).mean()

    df = pd.DataFrame({
        '总泡沫指数': total_smoothed,
        '综合情绪指标': sentiment_index,
        '综合资金指标': capital_index,
    }).dropna()

    # 清理时区并修改索引名称为中文，完美解决图表提示框中英混杂问题
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index.name = '日期'

    return df


# --- 网页渲染 ---
with st.spinner('📡 正在从华尔街同步底层数据并渲染全交互大屏...'):
    df = fetch_and_calculate()

st.sidebar.header("⚙️ 看板控制台")
days = st.sidebar.slider("时间轴范围 (天)", 100, 1500, 400)
plot_df = df.tail(days)

# --- 🚀 新增安全检查：防止周末API抽风导致数据为空 ---
if plot_df.empty:
    st.warning("⚠️ 暂未获取到最新行情数据。可能是周末雅虎财经 API 维护或云端网络延迟，请稍后再试。")
    st.stop()  # 停止向下运行，防止页面红屏报错

# 如果数据正常，则安全计算
val = plot_df['总泡沫指数'].iloc[-1]
# 加个双保险，防止数据只有1天导致拿不到昨天(-2)的数据
delta = val - plot_df['总泡沫指数'].iloc[-2] if len(plot_df) > 1 else 0

col1, col2, col3 = st.columns(3)
col1.metric("🚨 美投 AI 泡沫指数", f"{val:.1f}", f"{delta:.2f}", delta_color="inverse")

# 警戒线升级到 85
if val >= 85:
    status, color = "极度危险 (超越警戒线)", "🔴"
elif val >= 65:
    status, color = "偏高 (需警惕回调)", "🟠"
elif val >= 40:
    status, color = "中性 (健康状态)", "🟡"
else:
    status, color = "恐慌底部 (黄金坑)", "🟢"

col2.metric("📊 市场状态评级", f"{color} {status}")
col3.metric("📅 最新更新日期", plot_df.index[-1].strftime('%Y-%m-%d'))

st.subheader("🌐 综合指数走势 (警戒线: 85)")

# --- 使用 Plotly 构建全交互式主图 ---
fig_main = go.Figure()

# 1. 绘制主线
fig_main.add_trace(go.Scatter(
    x=plot_df.index,
    y=plot_df['总泡沫指数'],
    mode='lines',
    name='泡沫指数',
    line=dict(color='#004488', width=3.5),
    hovertemplate='日期: %{x|%Y-%m-%d}<br>指数数值: %{y:.2f}<extra></extra>'  # 纯中文提示框
))

# 2. 添加警戒线 (85) 和 中轴线 (50)
fig_main.add_hline(y=85, line_dash="solid", line_color="red", line_width=2, annotation_text="警戒线 (85)",
                   annotation_position="top left")
fig_main.add_hline(y=50, line_dash="dash", line_color="orange", annotation_text="中轴线 (50)",
                   annotation_position="top left")

# 3. 添加 85 以上的红色警报背景区域
fig_main.add_hrect(y0=85, y1=100, line_width=0, fillcolor="red", opacity=0.15)

# 4. 图表排版美化
fig_main.update_layout(
    height=450,
    margin=dict(l=0, r=0, t=10, b=0),
    yaxis=dict(range=[0, 100], gridcolor='rgba(0,0,0,0.1)'),
    xaxis=dict(gridcolor='rgba(0,0,0,0.1)'),
    plot_bgcolor='white',
    hovermode="x unified",  # 开启极具专业感的一体化悬停准星
    showlegend=False
)

# 渲染交互图表
st.plotly_chart(fig_main, use_container_width=True)

st.markdown("---")
st.subheader("🧩 核心指标拆解")
col_A, col_B = st.columns(2)
with col_A:
    st.markdown("**🧠 综合情绪指标 (占比 66.7%)**")
    st.line_chart(plot_df['综合情绪指标'], color='#1f77b4')
with col_B:
    st.markdown("**💰 综合资金指标 (占比 33.3%)**")
    st.line_chart(plot_df['综合资金指标'], color='#ff7f0e')
