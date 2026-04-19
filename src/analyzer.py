"""
src/analyzer.py — Phase 4, 5, 6: EDA, Feature Engineering & Analysis
======================================================================
All statistical analysis operations on cleaned expense DataFrames.
Used by both main.py (CLI) and app.py (Streamlit dashboard).
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional


class ExpenseAnalyzer:
    """
    Encapsulates all analysis operations for the Expense Tracker.

    Phases covered:
      Phase 4 — EDA          : shape, dtypes, distributions, null stats
      Phase 5 — Feature Eng  : rolling averages, Z-scores, budget flags
      Phase 6 — Analysis     : aggregations, trends, patterns
    """

    def __init__(self, df: pd.DataFrame):
        if df is None or df.empty:
            raise ValueError("ExpenseAnalyzer requires a non-empty DataFrame.")
        self.df = df.copy()
        self._validate()

    def _validate(self):
        required = {"amount", "category", "date", "month"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"DataFrame missing columns: {missing}")

    # ── Phase 4: EDA ─────────────────────────────────────────────────────────────
    def eda_summary(self) -> Dict:
        """Return a comprehensive EDA summary dictionary."""
        df = self.df
        return {
            "total_records"       : len(df),
            "date_range"          : (df["date"].min(), df["date"].max()),
            "total_spend"         : round(df["amount"].sum(), 2),
            "mean_transaction"    : round(df["amount"].mean(), 2),
            "median_transaction"  : round(df["amount"].median(), 2),
            "std_transaction"     : round(df["amount"].std(), 2),
            "min_transaction"     : round(df["amount"].min(), 2),
            "max_transaction"     : round(df["amount"].max(), 2),
            "unique_categories"   : df["category"].nunique(),
            "unique_months"       : df["month"].nunique(),
            "null_counts"         : df.isnull().sum().to_dict(),
            "category_value_counts": df["category"].value_counts().to_dict(),
        }

    def describe_amount(self) -> pd.DataFrame:
        """Pandas describe on amount column."""
        return self.df["amount"].describe().rename("Amount (₹)").to_frame()

    # ── Phase 5: Feature Engineering ─────────────────────────────────────────────
    def add_features(self) -> pd.DataFrame:
        """
        Add engineered features to the DataFrame.
        Returns the enriched DataFrame (does NOT modify self.df in-place).
        """
        df = self.df.copy()

        # Cumulative spending over time
        df = df.sort_values("date")
        df["cumulative_spend"] = df["amount"].cumsum()

        # 7-day rolling average spend (by date)
        daily = df.groupby("date")["amount"].sum().reset_index()
        daily["rolling_7d"] = daily["amount"].rolling(7, min_periods=1).mean().round(2)
        df = df.merge(daily[["date", "rolling_7d"]], on="date", how="left")

        # Z-score for anomaly flagging
        mean_amt = df["amount"].mean()
        std_amt  = df["amount"].std()
        df["z_score"]     = ((df["amount"] - mean_amt) / std_amt).round(3)
        df["is_anomaly"]  = df["z_score"].abs() > 2.5

        # Spend velocity (compared to category average)
        cat_avg = df.groupby("category")["amount"].transform("mean")
        df["vs_cat_avg_pct"] = (((df["amount"] - cat_avg) / cat_avg) * 100).round(1)

        return df

    # ── Phase 6: Analysis ─────────────────────────────────────────────────────────
    def category_totals(self) -> pd.DataFrame:
        """Category-wise total + % share."""
        totals = (
            self.df.groupby("category")["amount"]
            .agg(Total="sum", Count="count", Avg="mean")
            .round(2)
            .sort_values("Total", ascending=False)
            .reset_index()
        )
        totals["% Share"] = (totals["Total"] / totals["Total"].sum() * 100).round(1)
        return totals

    def monthly_trends(self) -> pd.DataFrame:
        """Month-over-month spending totals + MoM % change."""
        monthly = (
            self.df.groupby(["month", "month_dt"])["amount"]
            .agg(Total="sum", Transactions="count")
            .round(2)
            .reset_index()
            .sort_values("month_dt")
        )
        monthly["MoM_Change_%"] = monthly["Total"].pct_change().mul(100).round(1)
        return monthly

    def category_monthly_pivot(self) -> pd.DataFrame:
        """Pivot: rows=Month, cols=Category, values=Amount sum."""
        return (
            self.df
            .pivot_table(index="month", columns="category", values="amount",
                         aggfunc="sum", fill_value=0)
            .round(2)
        )

    def spending_by_weekday(self) -> pd.DataFrame:
        """Average spend per day of week."""
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        df = self.df.copy()
        day_avg = df.groupby("day_of_week")["amount"].mean().round(2).reset_index()
        day_avg.columns = ["Day", "Avg Spend"]
        day_avg["Day"] = pd.Categorical(day_avg["Day"], categories=order, ordered=True)
        return day_avg.sort_values("Day").reset_index(drop=True)

    def top_transactions(self, n: int = 10) -> pd.DataFrame:
        """Top N largest individual transactions."""
        return (
            self.df
            .nlargest(n, "amount")[["date", "category", "amount", "description"]]
            .reset_index(drop=True)
        )

    def payment_method_split(self) -> Optional[pd.DataFrame]:
        """Spending by payment method (if column exists)."""
        if "payment_method" not in self.df.columns:
            return None
        return (
            self.df.groupby("payment_method")["amount"]
            .agg(Total="sum", Count="count")
            .round(2)
            .sort_values("Total", ascending=False)
            .reset_index()
        )

    def weekend_vs_weekday(self) -> Dict:
        """Compare weekend vs weekday spending."""
        if "is_weekend" not in self.df.columns:
            return {}
        wk = self.df.groupby("is_weekend")["amount"].agg(["sum", "mean", "count"]).round(2)
        return {
            "Weekday Total" : float(wk.loc[False, "sum"])  if False in wk.index else 0,
            "Weekend Total" : float(wk.loc[True,  "sum"])  if True  in wk.index else 0,
            "Weekday Avg"   : float(wk.loc[False, "mean"]) if False in wk.index else 0,
            "Weekend Avg"   : float(wk.loc[True,  "mean"]) if True  in wk.index else 0,
        }

    def detect_subscription_creep(self) -> pd.DataFrame:
        """Flag categories whose monthly average is rising month-over-month."""
        monthly_cat = (
            self.df
            .groupby(["month_dt", "category"])["amount"]
            .sum()
            .reset_index()
            .sort_values("month_dt")
        )
        results = []
        for cat, grp in monthly_cat.groupby("category"):
            if len(grp) >= 2:
                slope = np.polyfit(range(len(grp)), grp["amount"].values, 1)[0]
                results.append({
                    "Category"      : cat,
                    "Monthly Slope ₹": round(slope, 2),
                    "Trend"         : "📈 Rising" if slope > 0 else "📉 Falling",
                })
        
        if not results:
            return pd.DataFrame(columns=["Category", "Monthly Slope ₹", "Trend"])
        
        return pd.DataFrame(results).sort_values("Monthly Slope ₹", ascending=False).reset_index(drop=True)

    def budget_performance(self, budgets: Dict[str, float]) -> pd.DataFrame:
        """
        Compare actual spending vs budget per category.

        Parameters
        ----------
        budgets : dict  e.g. {"Food & Dining": 5000, "Transport": 2000}
        """
        actual = self.df.groupby("category")["amount"].sum().round(2)
        rows = []
        for cat, budget in budgets.items():
            spent = actual.get(cat, 0.0)
            rows.append({
                "Category"    : cat,
                "Budget (₹)"  : budget,
                "Actual (₹)"  : spent,
                "Variance (₹)": round(spent - budget, 2),
                "% Used"      : round(spent / budget * 100, 1) if budget > 0 else 0,
                "Status"      : "🔴 Over" if spent > budget else "🟢 Under",
            })
        return pd.DataFrame(rows).sort_values("% Used", ascending=False).reset_index(drop=True)
