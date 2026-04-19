import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import time
from data_generator import generate_live_transaction
from ai_agents import AuraWatchAgent, AuraBrainAgent, AuraWellnessAgent, AuraGreenAgent

# -----------------------------------------------------------------------------
# Configuration & Ultra-Premium CSS
# -----------------------------------------------------------------------------
st.set_page_config(page_title="AuraFi Intelligence", page_icon="🏦", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
    
    .stApp {
        background-color: #F8FAFC !important;
        color: #0F172A !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Clean Top Padding */
    .block-container {
        padding-top: 1.5rem !important;
        max-width: 95% !important;
    }
    
    /* Glassmorphic Cards (Ultra Premium) */
    div[data-testid="stMetricValue"] {
        font-size: 2.4rem !important;
        font-weight: 800 !important;
        color: #1E293B !important;
        letter-spacing: -1px;
    }
    
    div[data-testid="stMetricLabel"] {
        font-size: 0.9rem !important;
        font-weight: 600 !important;
        text-transform: uppercase;
        letter-spacing: 1px;
        color: #64748B !important;
    }

    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.7);
        backdrop-filter: blur(20px);
        -webkit-backdrop-filter: blur(20px);
        border: 1px solid rgba(255, 255, 255, 0.5);
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.05), 0 8px 10px -6px rgba(0, 0, 0, 0.01);
        padding: 1.5rem;
        border-radius: 20px;
        transition: transform 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    
    div[data-testid="metric-container"]:hover {
        transform: translateY(-4px);
    }
    
    /* Main Gradient Header */
    .aura-header {
        font-size: 3rem;
        font-weight: 800;
        letter-spacing: -1.5px;
        background: linear-gradient(135deg, #2563EB 0%, #06B6D4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
    }
    
    /* Segmented Navigation Styling Override */
    .stTabs [data-baseweb="tab-list"] {
        background-color: transparent;
        gap: 2rem;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 1.1rem;
        font-weight: 600;
        color: #64748B;
        padding-bottom: 1rem;
    }
    .stTabs [aria-selected="true"] {
        color: #2563EB !important;
    }
    
    /* Action & Impact Cards */
    .impact-card {
        background: linear-gradient(135deg, #10B981 0%, #059669 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 15px -3px rgba(16, 185, 129, 0.3);
    }
    .crisis-card {
        background: linear-gradient(135deg, #EF4444 0%, #DC2626 100%);
        color: white;
        padding: 1.5rem;
        border-radius: 20px;
        box-shadow: 0 10px 15px -3px rgba(239, 68, 68, 0.3);
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Logic & Agents Context
# -----------------------------------------------------------------------------
@st.cache_resource
def load_agents():
    return AuraWatchAgent(), AuraBrainAgent(), AuraWellnessAgent(), AuraGreenAgent()

@st.cache_data
def load_historical_data():
    try:
        df = pd.read_csv("historical_transactions.csv")
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    except FileNotFoundError:
        st.error("Missing Data. Run data_generator.py")
        return pd.DataFrame()

watch, brain, wellness, green = load_agents()
df = load_historical_data()

if 'live_transactions' not in st.session_state:
    st.session_state.live_transactions = []
if 'anomalies_detected' not in st.session_state:
    st.session_state.anomalies_detected = []

if not watch.is_trained and not df.empty:
    watch.train(df)

# Top Bar
col_logo, col_space = st.columns([1, 10])
st.markdown('<div class="aura-header">AuraFi // V2 Ecosystem</div>', unsafe_allow_html=True)
st.markdown("**Autonomous Wealth, Global Impact & Financial Wellness** | 🟢 Live Feed Standby")

# Sidebar
st.sidebar.markdown('### Control Center')
sim_mode = st.sidebar.toggle("🌐 Enable Webhook Simulator")
st.sidebar.markdown("*(Injects live transactional telemetry)*")

if st.sidebar.button("Flush Cache"):
    st.session_state.live_transactions.clear()
    st.session_state.anomalies_detected.clear()

# Live Event Loop
if sim_mode:
    time.sleep(1.0)
    txn = generate_live_transaction()
    if watch.detect(txn):
        st.session_state.anomalies_detected.append(txn)
    st.session_state.live_transactions.insert(0, txn)
    if len(st.session_state.live_transactions) > 50:
        st.session_state.live_transactions = st.session_state.live_transactions[:50]

# Metrics Row
live_spend = sum([abs(t['Amount']) for t in st.session_state.live_transactions if t['Type'] == 'Expense'])
forecast = brain.forecast_end_of_month_expenses(df) if not df.empty else 0
health_score = wellness.calculate_score(df) if not df.empty else 50
carbon_footprint = green.analyze_total_footprint(df) if not df.empty else 0
roundups = sum([t['Charity_Roundup'] for t in st.session_state.live_transactions if 'Charity_Roundup' in t])

m1, m2, m3, m4, m5 = st.columns(5)
m1.metric("Live Session Spend", f"${live_spend:,.2f}")
m2.metric("Projected MoM Flow", f"${forecast:,.2f}", "+2.4% vs Avg")
m3.metric("Financial Wellness", f"{health_score}/100", "Stable", delta_color="normal")
m4.metric("CO2 Footprint (kg)", f"{carbon_footprint:,.1f}", "-5.0 kg optimized", delta_color="inverse")
m5.metric("Charity Roundups", f"${roundups:,.2f}", "Auto-Invested")

st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Multi-Hub Navigation
# -----------------------------------------------------------------------------
tab1, tab2, tab3 = st.tabs(["📊 Wealth Matrix", "🌍 The Green Ledger", "🧠 Debt-Trap AI"])

with tab1: # Wealth Matrix
    l_col, r_col = st.columns([6, 4])
    with l_col:
        st.subheader("Cash Trajectory")
        if not df.empty:
            timeline = df.copy()
            timeline.set_index('Date', inplace=True)
            inc = timeline[timeline['Type'] == 'Income'].resample('ME')['Amount'].sum()
            exp = timeline[timeline['Type'] == 'Expense'].resample('ME')['Amount'].sum().abs()
            
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=inc.index, y=inc.values, mode='lines', fill='tozeroy', name='Income', line=dict(color='#3B82F6', width=2)))
            fig.add_trace(go.Scatter(x=exp.index, y=exp.values, mode='lines', name='Expenses', line=dict(color='#EF4444', width=3)))
            fig.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)', margin=dict(l=0, r=0, t=10, b=0))
            st.plotly_chart(fig, use_container_width=True)
            
    with r_col:
        st.subheader("Live Telemetry")
        if st.session_state.live_transactions:
            live_df = pd.DataFrame(st.session_state.live_transactions)[['Vendor', 'Category', 'Amount', 'Carbon_Score']]
            live_df['Amount'] = live_df['Amount'].apply(lambda x: f"${abs(x):.2f}")
            st.dataframe(live_df, hide_index=True, use_container_width=True)
        else:
            st.info("Webhook offline. Enable via sidebar.")

        if st.session_state.anomalies_detected:
            st.error(f"🚨 {len(st.session_state.anomalies_detected)} Anomalies Detected by Aura-Watch!")

with tab2: # Green Ledger
    gl1, gl2 = st.columns([3, 7])
    with gl1:
        st.markdown(f"""
        <div class="impact-card">
            <h3>🌍 Planetary Impact</h3>
            <p>Every transaction is evaluated for estimated Carbon Cost based on vendor SEC data & category.</p>
            <h1 style="color:white; margin:0;">{carbon_footprint:,.0f} kg</h1>
            <p style="opacity: 0.8">YTD CO2 Footprint</p>
        </div>
        """, unsafe_allow_html=True)
        
    with gl2:
        if not df.empty:
            co2_df = df.groupby('Category')['Carbon_Score'].sum().sort_values(ascending=False).head(5)
            fig_bar = px.bar(co2_df, orientation='h', color_discrete_sequence=['#10B981'],
                             title="Top Emission Categories")
            fig_bar.update_layout(paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)')
            st.plotly_chart(fig_bar, use_container_width=True)

with tab3: # Debt Trap AI
    dw1, dw2 = st.columns([4, 6])
    with dw1:
        status, prescription, color = wellness.get_prescriptive_action(health_score)
        
        # 3D Premium Gauge
        fig_g = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = health_score,
            title = {'text': "Wellness Index"},
            gauge = {
                'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
                'bar': {'color': color},
                'steps': [
                    {'range': [0, 50], 'color': "rgba(239, 68, 68, 0.1)"},
                    {'range': [50, 80], 'color': "rgba(245, 158, 11, 0.1)"},
                    {'range': [80, 100], 'color': "rgba(16, 185, 129, 0.1)"}],
            }))
        fig_g.update_layout(paper_bgcolor='rgba(0,0,0,0)', font={'color': "#0F172A"})
        st.plotly_chart(fig_g, use_container_width=True)
        
    with dw2:
        st.markdown(f"""
        ### 🧠 Aura-Brain Prescription
        **Status:** {status}
        
        **Autonomous Action Plan:**
        - {prescription}
        - AI is monitoring your Discretionary bounds.
        - Next Auto-Sweep scheduled for Friday.
        """)
        
        if health_score < 60:
            st.markdown('<div class="crisis-card">⚠️ CRITICAL: Debt Spiral Protocol active. Recommend blocking Amazon & Uber APIs temporarily.</div>', unsafe_allow_html=True)

if sim_mode: st.rerun()
