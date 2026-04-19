"""
app.py — AuraFi Premium Expense Intelligence Platform
======================================================
Single, unified dashboard covering ALL 8 problem statement phases:
  Phase 2 — Data Input (CSV / Manual / Synthetic)
  Phase 3 — Cleaning (src.cleaner)
  Phase 4 — EDA Explorer
  Phase 5 — Feature Engineering (auto-applied)
  Phase 6 — Analysis Hub
  Phase 7 — Visualization Gallery
  Phase 8 — AI Insights Engine (AuraFi Agents)

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.ticker as mticker
import io, time, os
from datetime import datetime, timedelta
from pathlib import Path

# ── Local Modules ─────────────────────────────────────────────────────────────
from src.cleaner  import clean_dataframe
from src.analyzer import ExpenseAnalyzer
from src.insights import InsightsEngine
from data_generator import generate_historical_data
from ai_agents import AuraWatchAgent, AuraBrainAgent, AuraWellnessAgent, AuraGreenAgent

# ══════════════════════════════════════════════════════════════════════════════
# PAGE CONFIG
# ══════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="AuraFi | Expense Intelligence",
    page_icon="💎",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════════════════
# GLOBAL CSS — DARK GLASSMORPHIC "CYBER-FINTECH" THEME
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&family=Space+Grotesk:wght@400;500;600;700&display=swap');

  /* ── Base ── */
  html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
  }
  .stApp {
    background: linear-gradient(135deg, #0B0F1A 0%, #0D1629 40%, #0A1628 100%) !important;
    color: #E2E8F0 !important;
  }
  .block-container { padding-top: 1.2rem !important; max-width: 98% !important; }

  /* ── Scrollbar ── */
  ::-webkit-scrollbar { width: 6px; }
  ::-webkit-scrollbar-track { background: #0B0F1A; }
  ::-webkit-scrollbar-thumb { background: #1E3A5F; border-radius: 3px; }

  /* ── Sidebar ── */
  section[data-testid="stSidebar"] {
    background: rgba(11,15,26,0.95) !important;
    border-right: 1px solid rgba(0,240,255,0.12) !important;
  }
  section[data-testid="stSidebar"] * { color: #CBD5E1 !important; }

  /* ── Glass Card ── */
  .glass-card {
    background: rgba(255,255,255,0.04);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 20px;
    padding: 1.5rem 1.8rem;
    box-shadow: 0 8px 32px rgba(0,0,0,0.4), inset 0 1px 0 rgba(255,255,255,0.06);
    transition: transform 0.3s cubic-bezier(0.4,0,0.2,1), box-shadow 0.3s ease;
    margin-bottom: 1rem;
  }
  .glass-card:hover {
    transform: translateY(-3px);
    box-shadow: 0 16px 40px rgba(0,0,0,0.5), inset 0 1px 0 rgba(255,255,255,0.10);
  }

  /* ── Hero Header ── */
  .hero-header {
    background: linear-gradient(135deg, rgba(37,99,235,0.15) 0%, rgba(124,58,237,0.15) 100%);
    border: 1px solid rgba(37,99,235,0.3);
    border-radius: 24px;
    padding: 2.2rem 3rem;
    margin-bottom: 1.8rem;
    position: relative;
    overflow: hidden;
  }
  .hero-header::before {
    content: '';
    position: absolute; top: -50%; right: -10%;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(0,240,255,0.08) 0%, transparent 70%);
    border-radius: 50%;
  }
  .hero-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 3rem; font-weight: 800;
    background: linear-gradient(135deg, #00F0FF 0%, #2563EB 50%, #8A2BE2 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    letter-spacing: -1.5px; margin: 0;
  }
  .hero-sub {
    color: #94A3B8; font-size: 1rem; margin: 0.5rem 0 0;
    font-weight: 400; letter-spacing: 0.5px;
  }
  .hero-badge {
    display: inline-block;
    background: rgba(0,240,255,0.1);
    border: 1px solid rgba(0,240,255,0.3);
    color: #00F0FF; padding: 0.2rem 0.8rem;
    border-radius: 20px; font-size: 0.75rem; font-weight: 600;
    letter-spacing: 1px; margin-right: 0.5rem; margin-top: 0.8rem;
  }

  /* ── KPI Metric Cards ── */
  .kpi-grid { display: grid; grid-template-columns: repeat(5, 1fr); gap: 1rem; margin-bottom: 1.5rem; }
  .kpi-card {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 1.4rem 1.2rem;
    text-align: center;
    border-top: 3px solid;
    transition: transform .25s ease, box-shadow .25s ease;
    cursor: default;
  }
  .kpi-card:hover { transform: translateY(-4px); box-shadow: 0 12px 30px rgba(0,0,0,0.4); }
  .kpi-label {
    font-size: 0.7rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 1.5px; color: #64748B; margin-bottom: 0.5rem;
  }
  .kpi-value {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.9rem; font-weight: 800; color: #F1F5F9;
    letter-spacing: -1px;
  }
  .kpi-delta { font-size: 0.76rem; margin-top: 0.3rem; font-weight: 500; }

  /* ── Insight Cards ── */
  .insight-critical { background: rgba(239,68,68,0.08); border-left: 4px solid #EF4444; border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0; }
  .insight-warning  { background: rgba(245,158,11,0.08); border-left: 4px solid #F59E0B; border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0; }
  .insight-positive { background: rgba(16,185,129,0.08); border-left: 4px solid #10B981; border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0; }
  .insight-info     { background: rgba(37,99,235,0.08);  border-left: 4px solid #2563EB; border-radius: 12px; padding: 1rem 1.2rem; margin: 0.5rem 0; }
  .insight-title    { font-weight: 700; color: #F1F5F9; font-size: 0.95rem; margin-bottom: 0.3rem; }
  .insight-body     { color: #94A3B8; font-size: 0.87rem; line-height: 1.5; }

  /* ── Phase Badge ── */
  .phase-badge {
    display: inline-block; background: rgba(37,99,235,0.15);
    border: 1px solid rgba(37,99,235,0.4); color: #60A5FA;
    border-radius: 8px; padding: 0.15rem 0.6rem;
    font-size: 0.7rem; font-weight: 700; letter-spacing: 1px;
    text-transform: uppercase; margin-right: 0.5rem;
  }

  /* ── Section Title ── */
  .section-title {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 1.4rem; font-weight: 700; color: #F1F5F9;
    margin-bottom: 1rem; display: flex; align-items: center; gap: 0.6rem;
  }

  /* ── Tabs ── */
  .stTabs [data-baseweb="tab-list"] {
    background: transparent; gap: 0.3rem;
    border-bottom: 1px solid rgba(255,255,255,0.08);
  }
  .stTabs [data-baseweb="tab"] {
    font-family: 'Inter', sans-serif; font-weight: 600;
    font-size: 0.9rem; color: #64748B !important;
    padding: 0.6rem 1.2rem; border-radius: 10px 10px 0 0;
    transition: all .2s ease;
  }
  .stTabs [aria-selected="true"] {
    color: #00F0FF !important;
    background: rgba(0,240,255,0.06) !important;
    border-bottom: 2px solid #00F0FF !important;
  }

  /* ── Streamlit overrides ── */
  div[data-testid="stMetricValue"] {
    font-family: 'Space Grotesk',sans-serif; font-size: 2rem !important;
    font-weight: 800 !important; color: #F1F5F9 !important;
  }
  div[data-testid="stMetricLabel"] {
    font-size: 0.75rem !important; font-weight: 600 !important;
    text-transform: uppercase; letter-spacing: 1px; color: #64748B !important;
  }
  div[data-testid="metric-container"] {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 18px; padding: 1.2rem;
  }

  /* ── DataFrame ── */
  div[data-testid="stDataFrame"] {
    border-radius: 14px; overflow: hidden;
    border: 1px solid rgba(255,255,255,0.08);
  }

  /* ── AuraFi Live Feed pills ── */
  .live-pill {
    display: inline-block; padding: 0.15rem 0.5rem;
    border-radius: 6px; font-size: 0.75rem; font-weight: 600;
  }
  .live-anomaly { background: rgba(239,68,68,0.15); color: #F87171; border: 1px solid rgba(239,68,68,0.3); }
  .live-normal  { background: rgba(16,185,129,0.1); color: #34D399; border: 1px solid rgba(16,185,129,0.2); }

  /* ── Action / Crisis cards ── */
  .agent-card {
    background: linear-gradient(135deg, rgba(16,185,129,0.12) 0%, rgba(5,150,105,0.08) 100%);
    border: 1px solid rgba(16,185,129,0.25); border-radius: 16px;
    padding: 1.2rem 1.5rem; color: #D1FAE5;
  }
  .crisis-card {
    background: linear-gradient(135deg, rgba(239,68,68,0.12) 0%, rgba(220,38,38,0.08) 100%);
    border: 1px solid rgba(239,68,68,0.3); border-radius: 16px;
    padding: 1.2rem 1.5rem; color: #FCA5A5;
  }

  /* ── Budget Progress ── */
  progress { appearance: none; width: 100%; height: 8px; border-radius: 4px; overflow: hidden; }
  progress::-webkit-progress-bar { background: rgba(255,255,255,0.08); border-radius: 4px; }
  progress.good::-webkit-progress-value { background: #10B981; }
  progress.warn::-webkit-progress-value { background: #F59E0B; }
  progress.over::-webkit-progress-value { background: #EF4444; }

  /* ── Upload Zone ── */
  div[data-testid="stFileUploader"] {
    border: 2px dashed rgba(0,240,255,0.3) !important;
    border-radius: 16px !important; background: rgba(0,240,255,0.03) !important;
  }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# PLOTLY DARK THEME
# ══════════════════════════════════════════════════════════════════════════════
DARK_BG    = "rgba(0,0,0,0)"
GRID_COLOR = "rgba(255,255,255,0.06)"
TEXT_COLOR = "#94A3B8"
LINE_COLOR = "#1E293B"

PLOTLY_LAYOUT = dict(
    paper_bgcolor=DARK_BG, plot_bgcolor=DARK_BG,
    font=dict(family="Inter", color=TEXT_COLOR, size=12),
    margin=dict(l=10, r=10, t=40, b=10),
    xaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR, showline=False),
    yaxis=dict(gridcolor=GRID_COLOR, linecolor=LINE_COLOR, showline=False),
    legend=dict(bgcolor="rgba(0,0,0,0.3)", bordercolor="rgba(255,255,255,0.1)", borderwidth=1),
)

COLORS = {
    "blue"  : "#2563EB", "cyan"  : "#00F0FF", "purple": "#8A2BE2",
    "green" : "#10B981", "red"   : "#EF4444", "amber" : "#F59E0B",
    "pink"  : "#FF3366", "indigo": "#6366F1",
}
CAT_PALETTE = [
    "#2563EB","#00F0FF","#8A2BE2","#10B981","#F59E0B",
    "#EF4444","#6366F1","#EC4899","#14B8A6","#F97316",
]

# ══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ══════════════════════════════════════════════════════════════════════════════
@st.cache_resource(show_spinner=False)
def load_agents():
    return AuraWatchAgent(), AuraBrainAgent(), AuraWellnessAgent(), AuraGreenAgent()

@st.cache_data(show_spinner=False)
def load_synthetic() -> pd.DataFrame:
    raw = generate_historical_data(12)
    raw_e = raw[raw["Type"] == "Expense"].copy()
    raw_e = raw_e.rename(columns={"Date":"date","Category":"category","Amount":"amount","Vendor":"description"})
    return clean_and_enrich(raw_e)

def clean_and_enrich(raw_df: pd.DataFrame) -> pd.DataFrame:
    df = clean_dataframe(raw_df)
    try:
        az = ExpenseAnalyzer(df)
        df = az.add_features()
    except Exception:
        pass
    return df

def inr(v: float) -> str:
    return f"₹{v:,.0f}"

def plotly_fig(fig: go.Figure) -> go.Figure:
    fig.update_layout(**PLOTLY_LAYOUT)
    return fig

def phase_tag(n: int, label: str) -> str:
    return label

def insight_html(ins) -> str:
    cls_map = {"critical":"insight-critical","warning":"insight-warning","positive":"insight-positive","info":"insight-info"}
    cls = cls_map.get(ins.level, "insight-info")
    return f"""
    <div class="{cls}">
      <div class="insight-title">{ins.icon} {ins.title} <span style="font-size:0.72rem;color:#475569;font-weight:400">[{ins.category}]</span></div>
      <div class="insight-body">{ins.body}</div>
    </div>"""

# ══════════════════════════════════════════════════════════════════════════════
# AGENTS (Phase 8 / AuraFi Layer)
# ══════════════════════════════════════════════════════════════════════════════
watch, brain, wellness, green = load_agents()

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR — DATA INPUT (PHASE 2)
# ══════════════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("""
    <div style="text-align:center;padding:1rem 0 0.5rem">
      <span style="font-size:2.2rem">💎</span><br>
      <span style="font-family:'Space Grotesk';font-size:1.2rem;font-weight:800;
        background:linear-gradient(135deg,#00F0FF,#8A2BE2);
        -webkit-background-clip:text;-webkit-text-fill-color:transparent;">
        AuraFi
      </span><br>
      <span style="font-size:0.72rem;color:#475569;letter-spacing:2px;text-transform:uppercase;">
        Expense Intelligence
      </span>
    </div>
    <hr style="border-color:rgba(255,255,255,0.06);margin:0.8rem 0">
    """, unsafe_allow_html=True)

    st.markdown('<span class="phase-badge">Phase 2</span> **Data Source**', unsafe_allow_html=True)
    data_choice = st.radio("", ["🎲 Synthetic Data", "📂 Upload CSV", "✏️ Manual Entry"], label_visibility="collapsed")

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:0.8rem 0">', unsafe_allow_html=True)
    st.markdown('<span class="phase-badge">Phase 8</span> **AuraFi Live**', unsafe_allow_html=True)
    sim_mode = st.toggle("🔴 Enable Live Feed", value=False)
    if sim_mode:
        st.caption("*Injecting real-time transaction stream*")

    st.markdown('<hr style="border-color:rgba(255,255,255,0.06);margin:0.8rem 0">', unsafe_allow_html=True)
    st.markdown("**💳 Budget Settings**")
    budgets = {}
    with st.expander("Set category budgets", expanded=False):
        for cat in ["Food & Dining","Transport","Shopping","Entertainment","Healthcare","Subscriptions"]:
            budgets[cat] = st.number_input(cat, min_value=0, value=5000, step=500, key=f"b_{cat}")

# ══════════════════════════════════════════════════════════════════════════════
# DATA LOADING (PHASE 2 → 3 → 5)
# ══════════════════════════════════════════════════════════════════════════════
SAMPLE_CATS = ["Food & Dining","Transport","Housing","Utilities",
               "Shopping","Entertainment","Healthcare","Education","Subscriptions"]

df = None

if data_choice == "🎲 Synthetic Data":
    with st.spinner("Generating 12-month synthetic dataset..."):
        df = load_synthetic()

elif data_choice == "📂 Upload CSV":
    uploaded = st.sidebar.file_uploader("Upload CSV", type="csv",
        help="Required: Date, Category, Amount. Optional: Description, Payment_Method")
    st.sidebar.markdown("""
**CSV Format:**
```
Date,Category,Amount,Description
2024-01-05,Food & Dining,450,Lunch at Zomato
2024-01-08,Transport,120,Uber ride
```""")
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            df  = clean_and_enrich(raw)
            st.sidebar.success(f"✅ Loaded {len(df)} transactions")
        except Exception as e:
            st.sidebar.error(f"Error: {e}")

elif data_choice == "✏️ Manual Entry":
    st.sidebar.markdown("**Add Expense**")
    with st.sidebar.form("add_form", clear_on_submit=True):
        e_date = st.date_input("Date", datetime.today())
        e_cat  = st.selectbox("Category", SAMPLE_CATS)
        e_amt  = st.number_input("Amount (₹)", min_value=1.0, value=500.0, step=10.0)
        e_desc = st.text_input("Description", placeholder="e.g. Lunch, Uber")
        e_pay  = st.selectbox("Payment Method", ["UPI","Credit Card","Debit Card","Cash","Net Banking"])
        if st.form_submit_button("➕ Add"):
            if "manual_rows" not in st.session_state:
                st.session_state.manual_rows = []
            st.session_state.manual_rows.append({"date":str(e_date),"category":e_cat,
                "amount":e_amt,"description":e_desc,"payment_method":e_pay})
            st.sidebar.success("Added!")
    if st.sidebar.button("🗑️ Clear All"):
        st.session_state.manual_rows = []
    if "manual_rows" in st.session_state and st.session_state.manual_rows:
        df = clean_and_enrich(pd.DataFrame(st.session_state.manual_rows))
    else:
        st.sidebar.info("Add at least one expense ↑")

# ── Session state for live feed ──────────────────────────────────────────────
if "live_txns"    not in st.session_state: st.session_state.live_txns    = []
if "live_anomaly" not in st.session_state: st.session_state.live_anomaly = []

# ══════════════════════════════════════════════════════════════════════════════
# HERO HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero-header">
  <div class="hero-title">AuraFi Expense Intelligence</div>
  <div class="hero-sub">Multi-Agent AI · Real-Time Analytics · Autonomous Wealth Intelligence</div>
  <div style="margin-top:0.8rem">
    <span style="color:#64748B;font-size:0.8rem;font-weight:600;letter-spacing:1px">PROFESSIONAL DATA SCIENCE ECOSYSTEM</span>
  </div>
</div>
""", unsafe_allow_html=True)

if df is None or df.empty:
    st.info("👈 Choose a data source from the sidebar to begin the analysis pipeline.")
    st.stop()

# ── Train AuraWatch ──────────────────────────────────────────────────────────
if not watch.is_trained:
    try:
        hist = load_synthetic()
        watch.train(hist)
    except Exception:
        pass

# ── Live Feed ────────────────────────────────────────────────────────────────
if sim_mode:
    from data_generator import generate_live_transaction
    time.sleep(0.8)
    txn = generate_live_transaction()
    if watch.detect(txn):
        st.session_state.live_anomaly.insert(0, txn)
    st.session_state.live_txns.insert(0, txn)
    st.session_state.live_txns = st.session_state.live_txns[:40]

# ══════════════════════════════════════════════════════════════════════════════
# KPI ROW
# ══════════════════════════════════════════════════════════════════════════════
az = ExpenseAnalyzer(df)
eda = az.eda_summary()
monthly = az.monthly_trends()
cat_tot = az.category_totals()
health  = wellness.calculate_score(df.rename(columns={"amount":"Amount","category":"Category"}).assign(Type="Expense"))
co2     = green.analyze_total_footprint(df) if "carbon_score" in df.columns else round(df["amount"].sum() * 0.3, 1)
forecast= brain.forecast_end_of_month_expenses(df.rename(columns={"amount":"Amount","category":"Category","date":"Date","month":"YearMonth"}).assign(Type="Expense")) if "amount" in df.columns else 0

# MoM change
mom = monthly["MoM_Change_%"].iloc[-1] if len(monthly) > 1 and not pd.isna(monthly["MoM_Change_%"].iloc[-1]) else 0.0

k1,k2,k3,k4,k5 = st.columns(5)

def kpi(col, label, val, delta=None, color="#2563EB"):
    delta_html = f'<div class="kpi-delta" style="color:{"#10B981" if (delta or 0) < 0 or "↑" in str(delta) else "#EF4444"}">{delta}</div>' if delta else ""
    col.markdown(f"""
    <div class="kpi-card" style="border-top-color:{color}">
      <div class="kpi-label">{label}</div>
      <div class="kpi-value">{val}</div>
      {delta_html}
    </div>""", unsafe_allow_html=True)

kpi(k1, "Total Spent",    inr(eda["total_spend"]),  color="#2563EB")
kpi(k2, "Monthly Avg",    inr(eda["mean_transaction"] * 30), color="#8A2BE2")
kpi(k3, "MoM Change",     f"{mom:+.1f}%", color="#F59E0B" if abs(mom) > 15 else "#10B981")
kpi(k4, "Wellness Score", f"{health}/100", color="#10B981" if health > 70 else "#EF4444")
kpi(k5, "Transactions",   str(eda["total_records"]), color="#00F0FF")

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# TABS
# ══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "🔍 EDA Explorer",
    "📈 Trends & Patterns",
    "🎨 Visualization Gallery",
    "💡 AI Insights",
    "🤖 AuraFi Live",
])

# ════════════════════════════════════════
# TAB 1 — OVERVIEW (Phase 4 EDA + 6 Analysis)
# ════════════════════════════════════════
with tab1:
    st.markdown(f'<div class="section-title">{phase_tag(4,"EDA Summary")} {phase_tag(6,"Category Analysis")}</div>', unsafe_allow_html=True)

    c1, c2 = st.columns([6, 4])

    with c1:
        # Category bar chart
        fig_cat = go.Figure(go.Bar(
            x=cat_tot["Total"], y=cat_tot["category"],
            orientation="h",
            marker=dict(color=CAT_PALETTE[:len(cat_tot)],
                        line=dict(color="rgba(0,0,0,0)", width=0)),
            text=[f"₹{v:,.0f} ({p}%)" for v, p in zip(cat_tot["Total"], cat_tot["% Share"])],
            textposition="outside", textfont=dict(size=11, color="#94A3B8"),
        ))
        fig_cat.update_layout(**PLOTLY_LAYOUT,
            title="💰 Category-Wise Spending",
            title_font=dict(size=16, color="#F1F5F9", family="Space Grotesk"),
            height=360,
            bargap=0.3,
        )
        fig_cat.update_xaxes(tickformat="₹,.0f")
        st.plotly_chart(fig_cat, use_container_width=True)

    with c2:
        # Pie chart
        fig_pie = go.Figure(go.Pie(
            labels=cat_tot["category"],
            values=cat_tot["Total"],
            hole=0.55,
            marker=dict(colors=CAT_PALETTE[:len(cat_tot)], line=dict(color="#0B0F1A", width=2)),
            textinfo="percent",
            textfont=dict(size=11),
            hovertemplate="<b>%{label}</b><br>₹%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"₹{eda['total_spend']:,.0f}<br><span style='font-size:11px'>Total</span>",
                                font=dict(size=16, color="#F1F5F9", family="Space Grotesk"),
                                showarrow=False, x=0.5, y=0.5)
        fig_pie.update_layout(**{**PLOTLY_LAYOUT, "legend": dict(orientation="v", x=1, y=0.5)},
            title="🥧 Spend Distribution",
            title_font=dict(size=16, color="#F1F5F9", family="Space Grotesk"),
            height=360, showlegend=True,
        )
        st.plotly_chart(fig_pie, use_container_width=True)

    # Data table
    st.markdown(f'<div class="section-title" style="font-size:1rem">{phase_tag(6,"Category Summary Table")}</div>', unsafe_allow_html=True)
    display_cat = cat_tot.copy()
    display_cat["Total"] = display_cat["Total"].apply(inr)
    display_cat["Avg"]   = display_cat["Avg"].apply(inr)
    display_cat.columns  = ["Category","Total","Count","Avg Amount","% Share"]
    st.dataframe(display_cat, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 2 — EDA EXPLORER (Phase 4)
# ════════════════════════════════════════
with tab2:
    st.markdown(f'<div class="section-title">{phase_tag(4,"Exploratory Data Analysis")}</div>', unsafe_allow_html=True)

    # EDA stats
    col_a, col_b, col_c = st.columns(3)
    col_a.metric("Total Records",    eda["total_records"])
    col_b.metric("Date Range",       f'{eda["date_range"][0].strftime("%d %b")} → {eda["date_range"][1].strftime("%d %b %Y")}')
    col_c.metric("Unique Categories", eda["unique_categories"])

    col_d, col_e, col_f = st.columns(3)
    col_d.metric("Mean Transaction",   inr(eda["mean_transaction"]))
    col_e.metric("Median Transaction", inr(eda["median_transaction"]))
    col_f.metric("Std Deviation",      inr(eda["std_transaction"]))

    c3, c4 = st.columns(2)

    with c3:
        # Amount distribution histogram
        fig_hist = px.histogram(df, x="amount", nbins=35, color_discrete_sequence=["#8A2BE2"],
                                 marginal="box", labels={"amount":"Amount (₹)"},
                                 title="📈 Transaction Amount Distribution")
        fig_hist.update_traces(marker_line_color="#0B0F1A", marker_line_width=0.5, opacity=0.85)
        fig_hist.update_layout(**PLOTLY_LAYOUT, title_font=dict(size=15,color="#F1F5F9",family="Space Grotesk"), height=360)
        st.plotly_chart(fig_hist, use_container_width=True)

    with c4:
        # Box plot by category
        fig_box = px.box(df, x="category", y="amount", color="category",
                          color_discrete_sequence=CAT_PALETTE,
                          labels={"amount":"Amount (₹)", "category":"Category"},
                          title="📦 Spend Distribution by Category")
        fig_box.update_layout(**PLOTLY_LAYOUT, title_font=dict(size=15,color="#F1F5F9",family="Space Grotesk"),
                               height=360, showlegend=False, xaxis_tickangle=-30)
        st.plotly_chart(fig_box, use_container_width=True)

    # Full data table with filters
    st.markdown(f'<div class="section-title" style="font-size:1rem">{phase_tag(4,"Raw Transaction Data")}</div>', unsafe_allow_html=True)
    filter_cat = st.multiselect("Filter by Category", ["All"] + sorted(df["category"].unique().tolist()), default=["All"])
    df_display = df if "All" in filter_cat or not filter_cat else df[df["category"].isin(filter_cat)]

    show_cols = ["date","category","amount","description","month","spending_tier"]
    show_cols = [c for c in show_cols if c in df_display.columns]
    disp = df_display[show_cols].copy()
    disp["date"]   = disp["date"].dt.strftime("%d %b %Y")
    disp["amount"] = disp["amount"].apply(inr)
    st.dataframe(disp, use_container_width=True, hide_index=True, height=350)

    # CSV download
    csv_buf = df_display[show_cols].copy()
    csv_buf["date"] = csv_buf["date"].dt.strftime("%Y-%m-%d")
    st.download_button("⬇️ Download Filtered Data", csv_buf.to_csv(index=False).encode(),
                       "filtered_expenses.csv", "text/csv")

# ════════════════════════════════════════
# TAB 3 — TRENDS & PATTERNS (Phase 6)
# ════════════════════════════════════════
with tab3:
    st.markdown(f'<div class="section-title">{phase_tag(6,"Monthly Trends & Spending Patterns")}</div>', unsafe_allow_html=True)

    c5, c6 = st.columns(2)

    with c5:
        # Monthly line
        monthly_plot = az.monthly_trends()
        fig_line = go.Figure()
        fig_line.add_trace(go.Scatter(
            x=monthly_plot["month_dt"], y=monthly_plot["Total"],
            mode="lines+markers",
            line=dict(color="#00F0FF", width=3),
            marker=dict(size=8, color="#00F0FF", line=dict(color="#0B0F1A", width=2)),
            fill="tozeroy", fillcolor="rgba(0,240,255,0.05)",
            name="Monthly Spend",
            hovertemplate="<b>%{x|%b %Y}</b><br>₹%{y:,.0f}<extra></extra>",
        ))
        fig_line.update_layout(**PLOTLY_LAYOUT,
            title="📅 Monthly Total Spending",
            title_font=dict(size=15, color="#F1F5F9", family="Space Grotesk"), height=340)
        fig_line.update_yaxes(tickformat="₹,.0f")
        st.plotly_chart(fig_line, use_container_width=True)

    with c6:
        # Stacked bar
        pivot = az.category_monthly_pivot().reset_index()
        pivot_melt = pivot.melt(id_vars="month", var_name="Category", value_name="Amount")

        fig_stack = px.bar(pivot_melt, x="month", y="Amount", color="Category",
                            barmode="stack", color_discrete_sequence=CAT_PALETTE,
                            labels={"month":"Month","Amount":"Amount (₹)"},
                            title="📊 Category Breakdown by Month")
        fig_stack.update_layout(**PLOTLY_LAYOUT,
            title_font=dict(size=15,color="#F1F5F9",family="Space Grotesk"), height=340,
            xaxis_tickangle=-35, bargap=0.2)
        fig_stack.update_yaxes(tickformat="₹,.0f")
        st.plotly_chart(fig_stack, use_container_width=True)

    c7, c8 = st.columns(2)

    with c7:
        # Weekday pattern
        day_order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        wday = az.spending_by_weekday()
        colors_day = ["#EF4444" if d in ["Saturday","Sunday"] else "#2563EB" for d in wday["Day"]]
        fig_day = go.Figure(go.Bar(
            x=wday["Day"], y=wday["Avg Spend"],
            marker_color=colors_day, marker_line_color="#0B0F1A", marker_line_width=0.5,
            text=[inr(v) for v in wday["Avg Spend"]], textposition="outside",
            textfont=dict(size=9, color="#94A3B8"),
        ))
        fig_day.update_layout(**PLOTLY_LAYOUT,
            title="📆 Avg Spend by Day (🔴 Weekend)",
            title_font=dict(size=15,color="#F1F5F9",family="Space Grotesk"), height=300)
        fig_day.update_yaxes(tickformat="₹,.0f")
        st.plotly_chart(fig_day, use_container_width=True)

    with c8:
        # Subscription creep
        creep = az.detect_subscription_creep()
        col_map = creep["Trend"].map({"📈 Rising":"#EF4444","📉 Falling":"#10B981"}).tolist()
        fig_creep = go.Figure(go.Bar(
            x=creep["Monthly Slope ₹"], y=creep["Category"],
            orientation="h",
            marker_color=col_map,
            text=[f"₹{v:+.0f}/mo" for v in creep["Monthly Slope ₹"]],
            textposition="outside", textfont=dict(size=9, color="#94A3B8"),
        ))
        fig_creep.update_layout(**PLOTLY_LAYOUT,
            title="📈 Subscription Creep — Category Slope",
            title_font=dict(size=15,color="#F1F5F9",family="Space Grotesk"), height=300,
            xaxis_title="₹/Month")
        st.plotly_chart(fig_creep, use_container_width=True)

    # Budget performance
    if budgets:
        st.markdown(f'<div class="section-title" style="font-size:1rem">{phase_tag(6,"Budget vs Actual")}</div>', unsafe_allow_html=True)
        budget_df = az.budget_performance(budgets)
        for _, row in budget_df.iterrows():
            pct = min(row["% Used"], 100)
            bar_class = "over" if row["% Used"] >= 100 else ("warn" if row["% Used"] >= 80 else "good")
            color = "#EF4444" if row["% Used"] >= 100 else ("#F59E0B" if row["% Used"] >= 80 else "#10B981")
            st.markdown(f"""
            <div style="margin:0.6rem 0">
              <div style="display:flex;justify-content:space-between;margin-bottom:0.3rem">
                <span style="color:#E2E8F0;font-weight:600;font-size:0.9rem">{row['Category']}</span>
                <span style="color:{color};font-size:0.85rem;font-weight:600">{row['Status']} • {row['% Used']:.0f}% • {inr(row['Actual (₹)'])} / {inr(row['Budget (₹)'])}</span>
              </div>
              <div style="background:rgba(255,255,255,0.06);border-radius:4px;height:8px;overflow:hidden">
                <div style="width:{pct:.1f}%;height:100%;background:{color};border-radius:4px;transition:width 0.5s ease"></div>
              </div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 4 — VISUALIZATION GALLERY (Phase 7)
# ════════════════════════════════════════
with tab4:
    st.markdown(f'<div class="section-title">{phase_tag(7,"Full Visualization Gallery")}</div>', unsafe_allow_html=True)

    # Heatmap
    pivot_heat = az.category_monthly_pivot()
    fig_heat = px.imshow(pivot_heat,
                          color_continuous_scale="Blues",
                          labels=dict(x="Month", y="Category", color="₹ Spent"),
                          title="🔥 Spending Heatmap — Category × Month",
                          aspect="auto")
    fig_heat.update_layout(**PLOTLY_LAYOUT,
        title_font=dict(size=15,color="#F1F5F9",family="Space Grotesk"), height=400,
        coloraxis_colorbar=dict(tickformat="₹,.0f", title="Spend"))
    fig_heat.update_xaxes(tickangle=-35)
    st.plotly_chart(fig_heat, use_container_width=True)

    c9, c10 = st.columns(2)
    with c9:
        # Cumulative spend
        df_sorted = df.sort_values("date").copy()
        df_sorted["cumulative"] = df_sorted["amount"].cumsum()
        fig_cum = go.Figure()
        fig_cum.add_trace(go.Scatter(
            x=df_sorted["date"], y=df_sorted["cumulative"],
            mode="lines", line=dict(color="#10B981", width=2.5),
            fill="tozeroy", fillcolor="rgba(16,185,129,0.07)",
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Cumulative: ₹%{y:,.0f}<extra></extra>",
        ))
        fig_cum.update_layout(**PLOTLY_LAYOUT,
            title="📈 Cumulative Spending Over Time",
            title_font=dict(size=14,color="#F1F5F9",family="Space Grotesk"), height=330)
        fig_cum.update_yaxes(tickformat="₹,.0f")
        st.plotly_chart(fig_cum, use_container_width=True)

    with c10:
        # Anomaly scatter
        df_anom = df.copy() if "is_anomaly" not in df.columns else df.copy()
        if "is_anomaly" not in df_anom.columns:
            df_anom["is_anomaly"] = False
        fig_anom = px.scatter(df_anom, x="date", y="amount", color="is_anomaly",
                               color_discrete_map={False:"#3B82F6", True:"#EF4444"},
                               symbol="is_anomaly",
                               symbol_map={False:"circle", True:"x"},
                               size_max=12,
                               labels={"amount":"Amount (₹)","is_anomaly":"Anomaly"},
                               title="🔍 Anomaly Detection Scatter",
                               hover_data=["category"])
        fig_anom.update_traces(marker=dict(size=7, opacity=0.75))
        fig_anom.update_layout(**PLOTLY_LAYOUT,
            title_font=dict(size=14,color="#F1F5F9",family="Space Grotesk"), height=330)
        fig_anom.update_yaxes(tickformat="₹,.0f")
        st.plotly_chart(fig_anom, use_container_width=True)

    # Sunburst — spending breakdown (Category → Spending Tier)
    if "spending_tier" in df.columns:
        fig_sun = px.sunburst(df, path=["category","spending_tier"], values="amount",
                               color="amount", color_continuous_scale="Blues_r",
                               title="☀️ Hierarchical Spending: Category → Spend Tier")
        fig_sun.update_layout(**PLOTLY_LAYOUT,
            title_font=dict(size=14,color="#F1F5F9",family="Space Grotesk"), height=420)
        st.plotly_chart(fig_sun, use_container_width=True)

    # Top transactions
    st.markdown(f'<div class="section-title" style="font-size:1rem">{phase_tag(7,"Top 10 Largest Transactions")}</div>', unsafe_allow_html=True)
    top_txns = az.top_transactions(10)
    top_txns["date"]   = top_txns["date"].dt.strftime("%d %b %Y")
    top_txns["amount"] = top_txns["amount"].apply(inr)
    st.dataframe(top_txns, use_container_width=True, hide_index=True)

# ════════════════════════════════════════
# TAB 5 — AI INSIGHTS (Phase 8)
# ════════════════════════════════════════
with tab5:
    st.markdown(f'<div class="section-title">{phase_tag(8,"Automated AI Insights Engine")}</div>', unsafe_allow_html=True)

    engine  = InsightsEngine(df, budgets=budgets)
    all_ins = engine.run()

    # Summary row
    n_crit  = sum(1 for i in all_ins if i.level == "critical")
    n_warn  = sum(1 for i in all_ins if i.level == "warning")
    n_pos   = sum(1 for i in all_ins if i.level == "positive")
    n_info  = sum(1 for i in all_ins if i.level == "info")

    ia, ib, ic, id_ = st.columns(4)
    ia.markdown(f'<div style="text-align:center;padding:1rem;background:rgba(239,68,68,0.08);border-radius:14px;border:1px solid rgba(239,68,68,0.2)"><div style="font-size:1.8rem;font-weight:800;color:#EF4444">{n_crit}</div><div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:1px">Critical</div></div>', unsafe_allow_html=True)
    ib.markdown(f'<div style="text-align:center;padding:1rem;background:rgba(245,158,11,0.08);border-radius:14px;border:1px solid rgba(245,158,11,0.2)"><div style="font-size:1.8rem;font-weight:800;color:#F59E0B">{n_warn}</div><div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:1px">Warnings</div></div>', unsafe_allow_html=True)
    ic.markdown(f'<div style="text-align:center;padding:1rem;background:rgba(16,185,129,0.08);border-radius:14px;border:1px solid rgba(16,185,129,0.2)"><div style="font-size:1.8rem;font-weight:800;color:#10B981">{n_pos}</div><div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:1px">Positive</div></div>', unsafe_allow_html=True)
    id_.markdown(f'<div style="text-align:center;padding:1rem;background:rgba(37,99,235,0.08);border-radius:14px;border:1px solid rgba(37,99,235,0.2)"><div style="font-size:1.8rem;font-weight:800;color:#60A5FA">{n_info}</div><div style="font-size:0.75rem;color:#94A3B8;text-transform:uppercase;letter-spacing:1px">Info</div></div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Categorized insights
    categories = sorted(set(i.category for i in all_ins))
    sel_cat = st.selectbox("Filter by category", ["All"] + categories)
    filtered_ins = all_ins if sel_cat == "All" else [i for i in all_ins if i.category == sel_cat]

    for ins in filtered_ins:
        st.markdown(insight_html(ins), unsafe_allow_html=True)

    # AuraFi Agent Prescriptions
    st.markdown("<br>")
    st.markdown(f'<div class="section-title">{phase_tag(8,"AuraFi Multi-Agent Prescriptions")}</div>', unsafe_allow_html=True)

    status, prescription, color = wellness.get_prescriptive_action(health)
    d1, d2, d3 = st.columns([1, 2, 2])

    with d1:
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=health,
            title={"text":"Wellness Index","font":{"color":"#94A3B8","size":13}},
            number={"font":{"color":"#F1F5F9","size":40,"family":"Space Grotesk"},"suffix":"/100"},
            gauge={
                "axis": {"range":[None,100],"tickcolor":"#475569","tickwidth":1},
                "bar":  {"color":color},
                "steps":[
                    {"range":[0,50],"color":"rgba(239,68,68,0.08)"},
                    {"range":[50,80],"color":"rgba(245,158,11,0.08)"},
                    {"range":[80,100],"color":"rgba(16,185,129,0.08)"},
                ],
                "bordercolor":"rgba(255,255,255,0.05)",
            },
        ))
        fig_gauge.update_layout(paper_bgcolor=DARK_BG, font={"color":"#94A3B8"}, height=240, margin=dict(l=20,r=20,t=30,b=5))
        st.plotly_chart(fig_gauge, use_container_width=True)

    with d2:
        st.markdown(f"""
        <div class="agent-card">
          <div style="font-weight:700;font-size:1rem;margin-bottom:0.5rem">🧠 Aura-Brain Prescription</div>
          <div style="font-size:0.85rem;color:#D1FAE5;margin-bottom:0.4rem"><b>Status:</b> {status}</div>
          <div style="font-size:0.85rem;color:#D1FAE5">• {prescription}</div>
          <div style="font-size:0.82rem;color:#6EE7B7;margin-top:0.5rem">• Monitoring discretionary spend bounds</div>
          <div style="font-size:0.82rem;color:#6EE7B7">• Auto-Sweep recommendation scheduled</div>
        </div>""", unsafe_allow_html=True)

    with d3:
        if health < 60:
            st.markdown(f"""
            <div class="crisis-card">
              <div style="font-weight:700;margin-bottom:0.5rem">🚨 Debt-Trap Alert</div>
              <div style="font-size:0.85rem">High overspending probability detected.<br>
              Aura-Action recommends:<br>
              • Pause non-essential subscriptions<br>
              • Freeze Shopping & Entertainment for 14 days<br>
              • Redirect 20% of spend to emergency fund</div>
            </div>""", unsafe_allow_html=True)
        else:
            top_save_cat = cat_tot.iloc[0]["category"]
            top_save_amt = cat_tot.iloc[0]["Total"]
            st.markdown(f"""
            <div class="agent-card">
              <div style="font-weight:700;margin-bottom:0.5rem">✅ Financial Health Good</div>
              <div style="font-size:0.85rem">Aura-Action recommends:<br>
              • Consider investing surplus in index funds<br>
              • Review <b>{top_save_cat}</b> (₹{top_save_amt:,.0f}) for 10% reduction<br>
              • Enable Auto-Sweep to savings account</div>
            </div>""", unsafe_allow_html=True)

# ════════════════════════════════════════
# TAB 6 — AURA-FI LIVE (Phase 8 Advanced)
# ════════════════════════════════════════
with tab6:
    st.markdown(f'<div class="section-title">{phase_tag(8,"AuraFi Live Transaction Feed")}</div>', unsafe_allow_html=True)

    l1, l2, l3, l4 = st.columns(4)
    l1.metric("Live Session Spend",
              inr(sum(abs(t["Amount"]) for t in st.session_state.live_txns if t.get("Type") == "Expense")))
    l2.metric("Anomalies Detected",  len(st.session_state.live_anomaly))
    l3.metric("CO₂ Footprint",       f"{co2:,.1f} kg")
    l4.metric("Projected Month-End", inr(forecast))

    st.markdown("<br>", unsafe_allow_html=True)

    col_feed, col_alert = st.columns([6, 4])

    with col_feed:
        st.markdown("#### 📡 Live Transaction Stream")
        if st.session_state.live_txns:
            for txn in st.session_state.live_txns[:15]:
                is_a = txn.get("Is_Synthetic_Anomaly", False)
                pill = '<span class="live-pill live-anomaly">⚠ ANOMALY</span>' if is_a else '<span class="live-pill live-normal">● Normal</span>'
                st.markdown(f"""
                <div style="display:flex;justify-content:space-between;align-items:center;
                  padding:0.6rem 1rem;margin:0.3rem 0;
                  background:rgba(255,255,255,0.03);border-radius:10px;
                  border:1px solid rgba(255,255,255,{"0.15" if is_a else "0.06"})">
                  <span>{pill} <b style="color:#E2E8F0">{txn.get("Vendor","—")}</b></span>
                  <span style="color:#94A3B8;font-size:0.85rem">{txn.get("Category","—")}</span>
                  <span style="color:{"#EF4444" if is_a else "#10B981"};font-weight:700">₹{abs(txn.get("Amount",0)):,.2f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:4rem;color:#475569;border:1px dashed rgba(255,255,255,0.1);border-radius:16px">
              <div style="font-size:2rem">📡</div>
              <div style="margin-top:0.5rem">Enable <b>Live Feed</b> in the sidebar to start streaming</div>
            </div>""", unsafe_allow_html=True)

    with col_alert:
        st.markdown("#### 🚨 Aura-Watch Alerts")
        if st.session_state.live_anomaly:
            for a in st.session_state.live_anomaly[:8]:
                st.markdown(f"""
                <div class="crisis-card" style="padding:0.8rem 1rem;margin:0.4rem 0">
                  <b>{a.get("Vendor","Unknown")}</b><br>
                  <span style="font-size:0.82rem">{a.get("Category","—")}</span>
                  <span style="float:right;font-weight:800">₹{abs(a.get("Amount",0)):,.2f}</span>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div style="text-align:center;padding:3rem 1rem;color:#475569;
              border:1px dashed rgba(16,185,129,0.2);border-radius:16px">
              <div style="font-size:1.5rem">🛡️</div>
              <div style="margin-top:0.5rem;color:#10B981;font-size:0.9rem">No anomalies detected</div>
            </div>""", unsafe_allow_html=True)

        if st.button("🗑️ Clear Live Data", use_container_width=True):
            st.session_state.live_txns    = []
            st.session_state.live_anomaly = []
            st.rerun()

# ── Auto-rerun for live feed ──────────────────────────────────────────────────
if sim_mode:
    st.rerun()
