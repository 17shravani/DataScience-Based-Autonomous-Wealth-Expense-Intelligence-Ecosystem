"""
Basic Expense Tracker — Problem Statement Implementation
=========================================================
✅ Covers:
  - Input expense data (CSV upload or manual entry)
  - Clean & analyze data using Pandas
  - Charts using Matplotlib / Seaborn
  - Category-wise spending
  - Monthly trends
  - Key insights
  - Streamlit dashboard
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
import io
from datetime import datetime

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Expense Tracker | Basic Analytics",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Styling ────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

  .stApp { background: #F0F4FF; }

  .hero {
    background: linear-gradient(135deg, #2563EB 0%, #7C3AED 100%);
    border-radius: 20px;
    padding: 2.5rem 3rem;
    color: white;
    margin-bottom: 2rem;
  }
  .hero h1 { font-size: 2.6rem; font-weight: 800; margin: 0; letter-spacing: -1px; }
  .hero p  { font-size: 1.05rem; opacity: 0.85; margin: 0.4rem 0 0; }

  .kpi-card {
    background: white;
    border-radius: 16px;
    padding: 1.5rem;
    text-align: center;
    box-shadow: 0 4px 20px rgba(0,0,0,0.07);
    border-top: 4px solid;
    transition: transform .25s ease;
  }
  .kpi-card:hover { transform: translateY(-3px); }
  .kpi-card .label { font-size: 0.78rem; font-weight: 600; text-transform: uppercase; letter-spacing: 1px; color: #64748B; }
  .kpi-card .value { font-size: 2rem; font-weight: 800; color: #0F172A; margin: 0.3rem 0 0; }

  .insight-box {
    background: #EFF6FF;
    border-left: 5px solid #2563EB;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.96rem;
    color: #1E3A8A;
  }
  .warn-box {
    background: #FFF7ED;
    border-left: 5px solid #F59E0B;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin: 0.5rem 0;
    font-size: 0.96rem;
    color: #92400E;
  }

  div[data-testid="stDataFrame"] { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

sns.set_theme(style="whitegrid", palette="muted")
CHART_BG = "#FFFFFF"


# ── Helpers ────────────────────────────────────────────────────────────────────
SAMPLE_CATEGORIES = ["Food & Dining", "Transport", "Housing", "Utilities",
                     "Shopping", "Entertainment", "Healthcare", "Education", "Other"]

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardise the uploaded dataframe."""
    df.columns = [c.strip().title() for c in df.columns]
    df["Date"]     = pd.to_datetime(df["Date"], dayfirst=False, errors="coerce")
    df["Amount"]   = pd.to_numeric(df["Amount"], errors="coerce").abs()
    df             = df.dropna(subset=["Date", "Amount"])
    df["Category"] = df["Category"].fillna("Other").str.strip().str.title()
    df["Month"]    = df["Date"].dt.to_period("M").astype(str)
    df["Month_dt"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    return df.sort_values("Date").reset_index(drop=True)

def generate_insights(df: pd.DataFrame):
    insights, warnings = [], []
    top_cat  = df.groupby("Category")["Amount"].sum().idxmax()
    top_amt  = df.groupby("Category")["Amount"].sum().max()
    insights.append(f"💸 Your biggest spending category is **{top_cat}** (₹{top_amt:,.2f} total).")

    monthly = df.groupby("Month_dt")["Amount"].sum()
    if len(monthly) >= 2:
        pct = ((monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2]) * 100
        arrow = "📈 up" if pct > 0 else "📉 down"
        insights.append(f"📅 Spending {arrow} **{abs(pct):.1f}%** vs the previous month.")

    avg_txn = df["Amount"].mean()
    max_txn = df["Amount"].max()
    if max_txn > avg_txn * 3:
        warnings.append(f"⚠️ Largest transaction (₹{max_txn:,.2f}) is **{max_txn/avg_txn:.1f}×** your average — check for anomalies.")

    unique_cats = df["Category"].nunique()
    if unique_cats > 6:
        insights.append(f"🗂️ Spending is spread across **{unique_cats}** categories — good diversification.")

    return insights, warnings

def fig_to_streamlit(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=140, bbox_inches="tight",
                facecolor=CHART_BG, edgecolor="none")
    buf.seek(0)
    st.image(buf, use_container_width=True)
    plt.close(fig)


# ── Sidebar — data source ───────────────────────────────────────────────────────
st.sidebar.markdown("## 📂 Data Source")
data_choice = st.sidebar.radio("Choose input method:",
                               ["Upload CSV", "Manual Entry", "Use Sample Data"])

df = None

# ── 1. CSV Upload ────────────────────────────────────────────────────────────────
if data_choice == "Upload CSV":
    uploaded = st.sidebar.file_uploader(
        "Upload your expense CSV", type="csv",
        help="Required columns: Date, Category, Amount, Description")

    st.sidebar.markdown("""
**Expected CSV format:**
```
Date,Category,Amount,Description
2024-01-05,Food & Dining,120,Lunch
2024-01-08,Transport,45,Uber
```
""")
    if uploaded:
        try:
            raw = pd.read_csv(uploaded)
            df  = clean_data(raw)
            st.sidebar.success(f"✅ Loaded {len(df)} transactions.")
        except Exception as e:
            st.sidebar.error(f"Error reading file: {e}")

# ── 2. Manual Entry ──────────────────────────────────────────────────────────────
elif data_choice == "Manual Entry":
    st.sidebar.markdown("### ➕ Add Expense")
    with st.sidebar.form("add_expense", clear_on_submit=True):
        e_date = st.date_input("Date", value=datetime.today())
        e_cat  = st.selectbox("Category", SAMPLE_CATEGORIES)
        e_amt  = st.number_input("Amount (₹)", min_value=0.01, step=0.01)
        e_desc = st.text_input("Description")
        if st.form_submit_button("Add ➕"):
            new_row = {"Date": str(e_date), "Category": e_cat,
                       "Amount": e_amt, "Description": e_desc}
            if "manual_expenses" not in st.session_state:
                st.session_state.manual_expenses = []
            st.session_state.manual_expenses.append(new_row)
            st.sidebar.success("Added!")

    if st.sidebar.button("🗑️ Clear All"):
        st.session_state.manual_expenses = []

    if "manual_expenses" in st.session_state and st.session_state.manual_expenses:
        raw = pd.DataFrame(st.session_state.manual_expenses)
        df  = clean_data(raw)
    else:
        st.sidebar.info("Add at least one expense above.")

# ── 3. Sample Data ───────────────────────────────────────────────────────────────
else:
    np.random.seed(42)
    dates  = pd.date_range("2024-01-01", periods=120, freq="3D")
    cats   = np.random.choice(SAMPLE_CATEGORIES, size=120,
                               p=[0.25, 0.15, 0.12, 0.08, 0.15, 0.10, 0.07, 0.05, 0.03])
    amounts = np.random.uniform(50, 800, size=120).round(2)
    amounts[10], amounts[55] = 4500, 3200
    sample_df = pd.DataFrame({
        "Date"       : dates,
        "Category"   : cats,
        "Amount"     : amounts,
        "Description": [f"Expense #{i+1}" for i in range(120)],
    })
    df = clean_data(sample_df)
    st.sidebar.info("🎲 Using 120 synthetic sample transactions.")


# ── Hero ─────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
  <h1>💰 Expense Tracker</h1>
  <p>Clean · Analyze · Visualize — Basic Data Science Dashboard</p>
</div>
""", unsafe_allow_html=True)

if df is None or df.empty:
    st.info("👈 Choose a data source from the sidebar to get started.")
    st.stop()

# ── Sidebar — filters ────────────────────────────────────────────────────────────
st.sidebar.markdown("---")
st.sidebar.markdown("## 🔍 Filters")

all_cats  = ["All"] + sorted(df["Category"].unique().tolist())
sel_cat   = st.sidebar.multiselect("Categories", all_cats, default=["All"])
if "All" not in sel_cat and sel_cat:
    df = df[df["Category"].isin(sel_cat)]

months     = sorted(df["Month"].unique().tolist())
sel_months = st.sidebar.multiselect("Months", months, default=months)
if sel_months:
    df = df[df["Month"].isin(sel_months)]

# ── KPI Row ───────────────────────────────────────────────────────────────────────
total     = df["Amount"].sum()
monthly_m = df.groupby("Month")["Amount"].sum()
avg_month = monthly_m.mean()
max_month = monthly_m.idxmax() if not monthly_m.empty else "—"
n_txn     = len(df)

kpi_data = [
    ("Total Spent",    f"₹{total:,.0f}",    "#2563EB"),
    ("Monthly Avg",    f"₹{avg_month:,.0f}", "#7C3AED"),
    ("Highest Month",  str(max_month),        "#059669"),
    ("Transactions",   str(n_txn),            "#F59E0B"),
]

cols = st.columns(4)
for col, (label, value, color) in zip(cols, kpi_data):
    col.markdown(f"""
    <div class="kpi-card" style="border-top-color:{color}">
      <div class="label">{label}</div>
      <div class="value">{value}</div>
    </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── [A] Category-Wise Spending ────────────────────────────────────────────────────
st.markdown("## 📊 Category-Wise Spending")
c1, c2 = st.columns(2)

cat_totals = df.groupby("Category")["Amount"].sum().sort_values(ascending=False)

with c1:
    st.markdown("### Bar Chart")
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=CHART_BG)
    palette = sns.color_palette("Blues_d", len(cat_totals))
    bars = ax.barh(cat_totals.index[::-1], cat_totals.values[::-1], color=palette)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.set_xlabel("Total Spent", fontsize=10)
    ax.set_title("Spending by Category", fontweight="bold", fontsize=12)
    ax.bar_label(bars, labels=[f"₹{v:,.0f}" for v in cat_totals.values[::-1]],
                 padding=4, fontsize=8.5)
    ax.set_facecolor(CHART_BG)
    fig.tight_layout()
    fig_to_streamlit(fig)

with c2:
    st.markdown("### Pie Chart")
    fig, ax = plt.subplots(figsize=(6, 4.5), facecolor=CHART_BG)
    colors = sns.color_palette("Set2", len(cat_totals))
    wedges, texts, autotexts = ax.pie(
        cat_totals.values,
        labels=cat_totals.index,
        autopct="%1.1f%%",
        pctdistance=0.78,
        colors=colors,
        startangle=140,
        wedgeprops=dict(edgecolor="white", linewidth=2),
    )
    for at in autotexts:
        at.set_fontsize(8)
        at.set_color("white")
        at.set_fontweight("bold")
    ax.set_title("Share of Total Spending", fontweight="bold", fontsize=12)
    ax.set_facecolor(CHART_BG)
    fig.tight_layout()
    fig_to_streamlit(fig)

# ── [B] Monthly Trends ────────────────────────────────────────────────────────────
st.markdown("## 📅 Monthly Spending Trends")

monthly_df  = df.groupby(["Month_dt", "Category"])["Amount"].sum().reset_index()
monthly_tot = df.groupby("Month_dt")["Amount"].sum().reset_index()

c3, c4 = st.columns(2)

with c3:
    st.markdown("### Total Monthly Spend")
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=CHART_BG)
    ax.plot(monthly_tot["Month_dt"], monthly_tot["Amount"],
            marker="o", linewidth=2.5, markersize=7,
            color="#2563EB", markerfacecolor="white", markeredgewidth=2)
    ax.fill_between(monthly_tot["Month_dt"], monthly_tot["Amount"],
                    alpha=0.15, color="#2563EB")
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}"))
    ax.set_title("Monthly Expenditure Over Time", fontweight="bold", fontsize=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount (₹)")
    plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.set_facecolor(CHART_BG)
    fig.tight_layout()
    fig_to_streamlit(fig)

with c4:
    st.markdown("### Category Stacked Trend")
    pivot = monthly_df.pivot_table(index="Month_dt", columns="Category",
                                    values="Amount", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=CHART_BG)
    palette_stacked = sns.color_palette("tab10", len(pivot.columns))
    pivot.plot(kind="bar", stacked=True, ax=ax, color=palette_stacked,
               edgecolor="white", linewidth=0.4)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda y, _: f"₹{y:,.0f}"))
    ax.set_title("Category Breakdown by Month", fontweight="bold", fontsize=12)
    ax.set_xlabel("Month"); ax.set_ylabel("Amount (₹)")
    ax.legend(loc="upper right", fontsize=7, ncol=2, framealpha=0.7)
    plt.xticks(rotation=35, ha="right", fontsize=8)
    ax.set_facecolor(CHART_BG)
    fig.tight_layout()
    fig_to_streamlit(fig)

# ── [C] Distribution ──────────────────────────────────────────────────────────────
st.markdown("## 🔍 Transaction Distribution")
c5, c6 = st.columns(2)

with c5:
    st.markdown("### Amount Distribution (Histogram)")
    fig, ax = plt.subplots(figsize=(7, 4), facecolor=CHART_BG)
    sns.histplot(df["Amount"], bins=25, kde=True, color="#7C3AED",
                 line_kws={"linewidth": 2}, ax=ax)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}"))
    ax.set_title("How Much Do You Spend Per Transaction?", fontweight="bold", fontsize=11)
    ax.set_xlabel("Amount (₹)"); ax.set_ylabel("Count")
    ax.set_facecolor(CHART_BG)
    fig.tight_layout()
    fig_to_streamlit(fig)

with c6:
    st.markdown("### Spending Heatmap (Month × Category)")
    pivot_heat = df.pivot_table(index="Category", columns="Month",
                                 values="Amount", aggfunc="sum", fill_value=0)
    fig, ax = plt.subplots(figsize=(max(7, len(pivot_heat.columns)*0.9), 4), facecolor=CHART_BG)
    sns.heatmap(pivot_heat, annot=True, fmt=".0f", cmap="YlOrRd",
                linewidths=0.5, linecolor="white", ax=ax,
                annot_kws={"size": 8})
    ax.set_title("Spending Heatmap", fontweight="bold", fontsize=12)
    ax.set_xlabel(""); ax.set_ylabel("")
    plt.xticks(rotation=45, ha="right", fontsize=8)
    ax.set_facecolor(CHART_BG)
    fig.tight_layout()
    fig_to_streamlit(fig)

# ── [D] Insights ─────────────────────────────────────────────────────────────────
st.markdown("## 💡 Key Insights")
insights, warnings = generate_insights(df)

icol, wcol = st.columns(2)
with icol:
    st.markdown("### 📌 Observations")
    for ins in insights:
        st.markdown(f'<div class="insight-box">{ins}</div>', unsafe_allow_html=True)

with wcol:
    st.markdown("### ⚠️ Flags")
    if warnings:
        for w in warnings:
            st.markdown(f'<div class="warn-box">{w}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="insight-box">✅ No anomalies detected in your spending patterns.</div>',
                    unsafe_allow_html=True)

# ── [E] Raw Data Table ────────────────────────────────────────────────────────────
st.markdown("## 📋 Raw Transaction Data")
display_df = df[["Date", "Category", "Amount", "Description", "Month"]].copy()
display_df["Date"]   = display_df["Date"].dt.strftime("%d %b %Y")
display_df["Amount"] = display_df["Amount"].apply(lambda x: f"₹{x:,.2f}")
st.dataframe(display_df.rename(columns={"Month": "Period"}),
             use_container_width=True, hide_index=True)

# Export button
csv_out = df[["Date", "Category", "Amount", "Description"]].copy()
csv_out["Date"] = csv_out["Date"].dt.strftime("%Y-%m-%d")
st.download_button("⬇️ Download Cleaned Data as CSV",
                   csv_out.to_csv(index=False).encode(),
                   "cleaned_expenses.csv", "text/csv")

# ── Footer ────────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#94A3B8;font-size:0.85rem;'>"
    "Basic Expense Tracker · Built with Pandas, Matplotlib, Seaborn & Streamlit"
    "</p>", unsafe_allow_html=True
)
