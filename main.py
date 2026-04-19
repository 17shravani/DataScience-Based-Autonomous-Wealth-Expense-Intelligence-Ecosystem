# -*- coding: utf-8 -*-
import sys, io
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

"""
main.py -- Complete Data Science Pipeline (CLI)
===============================================
Expense Tracker App | Phase 1 → Phase 9

Run: python main.py

Phases:
  Phase 1 — Setup & Configuration
  Phase 2 — Synthetic Data Generation
  Phase 3 — Data Cleaning
  Phase 4 — Exploratory Data Analysis (EDA)
  Phase 5 — Feature Engineering
  Phase 6 — Statistical Analysis
  Phase 7 — Visualization (saves to outputs/)
  Phase 8 — Automated Insights
  Phase 9 — Export (data/ + outputs/)
"""

import sys
import os
import time
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

# ── PHASE 1: Setup ──────────────────────────────────────────────────────────────
SEP  = "=" * 65
SEP2 = "-" * 65

def print_banner():
    print("\n" + SEP)
    print("  EXPENSE TRACKER APP -- Data Science Pipeline")
    print("  AuraFi Enhanced Edition")
    print("  " + datetime.now().strftime("%d %b %Y, %H:%M"))
    print(SEP + "\n")

def phase_header(n: int, title: str):
    print(f"\n{SEP2}")
    print(f"  Phase {n} >  {title}")
    print(SEP2)

print_banner()

# ─ Ensure output folders exist ──────────────────────────────────────────────────
phase_header(1, "Setup & Configuration")
for folder in ["data", "outputs", "images", "notebooks", "src"]:
    Path(folder).mkdir(exist_ok=True)
print("  [OK] Folder structure verified: data/ outputs/ images/ notebooks/ src/")

# ─ Install check ────────────────────────────────────────────────────────────────
try:
    import matplotlib, seaborn, sklearn
    print("  [OK] All dependencies present.")
except ImportError as e:
    print(f"  [ERR] Missing dependency: {e}")
    print("  Run: pip install -r requirements.txt")
    sys.exit(1)

# ── PHASE 2: Synthetic Data Generation ──────────────────────────────────────────
phase_header(2, "Synthetic Data Generation")

# Import the existing data_generator module
try:
    from data_generator import generate_historical_data
    print("  [INFO] Using existing data_generator.py (AuraFi synthetic engine)")
    df_raw = generate_historical_data(months=12)
    print(f"  [OK] Generated {len(df_raw)} synthetic transactions (12 months)")
except ImportError:
    print("  [INFO] Falling back to built-in data generator")
    # Built-in fallback
    np.random.seed(42)
    categories = ["Food & Dining","Transport","Housing","Utilities",
                  "Shopping","Entertainment","Healthcare","Education","Subscriptions"]
    dates  = pd.date_range("2024-01-01", "2024-12-31", freq="2D")
    n      = len(dates) * 2
    df_raw = pd.DataFrame({
        "Date"       : np.random.choice(dates, n),
        "Category"   : np.random.choice(categories, n, p=[.2,.15,.1,.08,.2,.1,.07,.05,.05]),
        "Amount"     : np.abs(np.random.normal(500, 300, n)).round(2),
        "Description": [f"Txn #{i+1}" for i in range(n)],
        "Type"       : "Expense",
    })
    print(f"  [OK] Generated {len(df_raw)} synthetic transactions (built-in)")

# Save raw data
raw_path = "data/raw_expenses.csv"
df_raw.to_csv(raw_path, index=False)
print(f"  [SAVED] Raw data -> {raw_path}")

# ── PHASE 3: Data Cleaning ───────────────────────────────────────────────────────
phase_header(3, "Data Cleaning")

from src.cleaner import clean_dataframe

# Map column names from data_generator output
df_for_cleaning = df_raw.rename(columns={
    "Date"    : "date",
    "Category": "category",
    "Amount"  : "amount",
    "Vendor"  : "description",
}).copy()

# Keep only expense rows if Type column exists
if "Type" in df_for_cleaning.columns:
    df_for_cleaning = df_for_cleaning[df_for_cleaning["Type"] == "Expense"].copy()

df_clean = clean_dataframe(df_for_cleaning)

clean_path = "data/clean_expenses.csv"
df_clean.to_csv(clean_path, index=False)
print(f"  [SAVED] Clean data -> {clean_path}")

# ── PHASE 4: Exploratory Data Analysis ──────────────────────────────────────────
phase_header(4, "Exploratory Data Analysis (EDA)")

from src.analyzer import ExpenseAnalyzer

analyzer = ExpenseAnalyzer(df_clean)
summary  = analyzer.eda_summary()

print(f"  Total Transactions  : {summary['total_records']}")
print(f"  Date Range          : {summary['date_range'][0].date()} to {summary['date_range'][1].date()}")
print(f"  Total Spend         : Rs {summary['total_spend']:,.2f}")
print(f"  Mean Transaction    : Rs {summary['mean_transaction']:,.2f}")
print(f"  Median Transaction  : Rs {summary['median_transaction']:,.2f}")
print(f"  Categories          : {summary['unique_categories']}")
print(f"  Months              : {summary['unique_months']}")

print("\n  📋 Describe Amount:")
print(analyzer.describe_amount().to_string(float_format=lambda x: f"₹{x:,.2f}"))

# ── PHASE 5: Feature Engineering ────────────────────────────────────────────────
phase_header(5, "Feature Engineering")

df_features = analyzer.add_features()
print("  [OK] Added features:")
print("     * cumulative_spend  -- running total over time")
print("     * rolling_7d        -- 7-day rolling average")
print("     * z_score           -- statistical anomaly score")
print("     * is_anomaly        -- True if |z_score| > 2.5")
print("     * vs_cat_avg_pct    -- % deviation from category average")
n_anomalies = df_features["is_anomaly"].sum()
print(f"  Anomalies detected  : {n_anomalies} transactions (|z| > 2.5)")

# ── PHASE 6: Statistical Analysis ───────────────────────────────────────────────
phase_header(6, "Statistical Analysis")

print("\n  📊 Category Totals:")
cat_df = analyzer.category_totals()
print(cat_df.to_string(index=False))

print("\n  📅 Monthly Trends:")
monthly_df = analyzer.monthly_trends()
print(monthly_df[["month","Total","MoM_Change_%"]].to_string(index=False))

print("\n  📆 Weekend vs Weekday:")
wk = analyzer.weekend_vs_weekday()
for k, v in wk.items():
    print(f"     {k}: ₹{v:,.2f}")

print("\n  🔍 Subscription Creep:")
creep = analyzer.detect_subscription_creep()
print(creep.to_string(index=False))

# Budget check (example budgets)
BUDGETS = {
    "Food & Dining" : 8000,
    "Transport"     : 4000,
    "Shopping"      : 6000,
    "Entertainment" : 3000,
    "Healthcare"    : 2000,
}
print("\n  💳 Budget Performance:")
budget_df = analyzer.budget_performance(BUDGETS)
print(budget_df.to_string(index=False))

# ── PHASE 7: Visualization ───────────────────────────────────────────────────────
phase_header(7, "Visualization (Generating Charts → outputs/)")

from src.visualizer import ExpenseVisualizer

# Use feature-engineered df (has is_anomaly)
viz = ExpenseVisualizer(df_features, theme="light")

charts = [
    ("Category Bar Chart",         viz.category_bar),
    ("Category Pie Chart",         viz.category_pie),
    ("Monthly Trend Line",         viz.monthly_trend),
    ("Stacked Category Monthly",   viz.stacked_category_monthly),
    ("Spending Heatmap",           viz.heatmap),
    ("Amount Distribution",        viz.amount_histogram),
    ("Weekday Pattern",            viz.weekday_bar),
    ("Cumulative Spend",           viz.cumulative_spend),
    ("Anomaly Scatter",            viz.anomaly_scatter),
    ("Summary Dashboard",          viz.summary_dashboard),
]

saved_charts = []
for name, fn in charts:
    try:
        _, path = fn()
        saved_charts.append(path)
        print(f"  [OK] {name:<35} -> {path}")
    except Exception as e:
        print(f"  [WARN] {name}: {e}")

# ── PHASE 8: Automated Insights ─────────────────────────────────────────────────
phase_header(8, "Automated Insights Generation")

from src.insights import InsightsEngine, format_insight_for_cli

engine   = InsightsEngine(df_features, budgets=BUDGETS)
insights = engine.run()

print(f"  Generated {len(insights)} financial insights:\n")
for insight in insights:
    print(format_insight_for_cli(insight))

# ── PHASE 9: Export / GitHub Ready ──────────────────────────────────────────────
phase_header(9, "Export & GitHub Preparation")

# Save analysis reports
cat_df.to_csv("outputs/category_analysis.csv", index=False)
monthly_df.to_csv("outputs/monthly_trends.csv", index=False)
budget_df.to_csv("outputs/budget_performance.csv", index=False)
df_features[df_features["is_anomaly"]].to_csv("outputs/anomalies.csv", index=False)

print("  [SAVED] Reports:")
print("     outputs/category_analysis.csv")
print("     outputs/monthly_trends.csv")
print("     outputs/budget_performance.csv")
print("     outputs/anomalies.csv")
print(f"     {len(saved_charts)} charts saved to outputs/")

# -- COMPLETE ------------------------------------------------------------------
print("\n" + SEP)
print("  PIPELINE COMPLETE -- All 9 Phases Done!")
print("  Run the premium dashboard: streamlit run app.py")
print("  GitHub: push data/, outputs/, src/ to your repo")
print(SEP + "\n")
