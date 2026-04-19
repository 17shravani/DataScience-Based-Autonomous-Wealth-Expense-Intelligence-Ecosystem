"""
src/insights.py — Phase 8: Automated Insights Engine
=====================================================
Generates human-readable, rule-based financial insights
from analyzed expense data. Usable in both CLI and Streamlit.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Tuple


# ── Insight dataclass substitute ────────────────────────────────────────────────
class Insight:
    """A single financial insight with severity level."""

    LEVELS = {"info": "[i]", "warning": "[!]", "critical": "[X]", "positive": "[OK]"}

    def __init__(self, title: str, body: str, level: str = "info", category: str = "General"):
        self.title    = title
        self.body     = body
        self.level    = level
        self.category = category
        self.icon     = self.LEVELS.get(level, "i")

    def __repr__(self):
        return f"{self.icon} [{self.level.upper()}] {self.title}: {self.body}"


# ── Main Insights Engine ─────────────────────────────────────────────────────────
class InsightsEngine:
    """
    Generates a ranked list of Insight objects from
    clean + analyzed expense DataFrames.
    """

    def __init__(self, df: pd.DataFrame, budgets: Dict[str, float] = None):
        self.df      = df.copy()
        self.budgets = budgets or {}

    def run(self) -> List[Insight]:
        """Run all insight generators and return a list of Insight objects."""
        insights: List[Insight] = []
        insights += self._top_category_insight()
        insights += self._monthly_trend_insight()
        insights += self._anomaly_insight()
        insights += self._weekend_spend_insight()
        insights += self._budget_insight()
        insights += self._savings_rate_insight()
        insights += self._subscription_creep_insight()
        insights += self._category_diversity_insight()
        # Sort: critical → warning → info → positive
        order = {"critical": 0, "warning": 1, "info": 2, "positive": 3}
        return sorted(insights, key=lambda x: order.get(x.level, 4))

    # ── Individual Insight Generators ────────────────────────────────────────────
    def _top_category_insight(self) -> List[Insight]:
        cat = self.df.groupby("category")["amount"].sum()
        top = cat.idxmax()
        pct = (cat.max() / cat.sum() * 100)
        level = "warning" if pct > 40 else "info"
        return [Insight(
            f"Top Spending: {top}",
            f"'{top}' consumes {pct:.1f}% of total spend (Rs {cat.max():,.0f}). "
            + ("Consider reviewing recurring costs here." if pct > 40 else "This is within healthy range."),
            level, "Spending Pattern"
        )]

    def _monthly_trend_insight(self) -> List[Insight]:
        monthly = self.df.groupby("month_dt")["amount"].sum().sort_index()
        if len(monthly) < 2:
            return []
        pct = (monthly.iloc[-1] - monthly.iloc[-2]) / monthly.iloc[-2] * 100
        if pct > 20:
            return [Insight("Spike Alert", f"Spending jumped {pct:+.1f}% this month vs last. Investigate unusual expenses.", "critical", "Monthly Trend")]
        elif pct > 10:
            return [Insight("Rising Spend", f"Monthly spend increased by {pct:+.1f}%. Monitor closely.", "warning", "Monthly Trend")]
        elif pct < -10:
            return [Insight("Spend Reduction", f"Great job! Spending fell {pct:.1f}% vs last month.", "positive", "Monthly Trend")]
        return [Insight("Stable Spending", f"Monthly spend change: {pct:+.1f}%. On track.", "info", "Monthly Trend")]

    def _anomaly_insight(self) -> List[Insight]:
        if "is_anomaly" not in self.df.columns:
            return []
        anomalies = self.df[self.df["is_anomaly"] == True]
        if anomalies.empty:
            return [Insight("No Anomalies", "All transactions are within expected spending ranges.", "positive", "Risk")]
        top_a = anomalies.nlargest(1, "amount").iloc[0]
        return [Insight(
            f"{len(anomalies)} Anomaly Transaction(s)",
            f"Largest outlier: Rs {top_a['amount']:,.0f} on {top_a['date'].strftime('%d %b %Y')} ({top_a['category']}). These may be one-time events or errors.",
            "warning", "Risk"
        )]

    def _weekend_spend_insight(self) -> List[Insight]:
        if "is_weekend" not in self.df.columns:
            return []
        weekend_avg = self.df[self.df["is_weekend"]]["amount"].mean()
        weekday_avg = self.df[~self.df["is_weekend"]]["amount"].mean()
        if weekday_avg == 0:
            return []
        ratio = weekend_avg / weekday_avg
        if ratio > 1.5:
            return [Insight(
                "Weekend Overspending",
                f"You spend {ratio:.1f}x more on weekends (avg Rs {weekend_avg:,.0f}) vs weekdays (Rs {weekday_avg:,.0f}). Consider setting a weekend budget.",
                "warning", "Behavior"
            )]
        return [Insight("Balanced Weekly Spend", "Weekday vs weekend spending is well-balanced.", "positive", "Behavior")]

    def _budget_insight(self) -> List[Insight]:
        if not self.budgets:
            return []
        results = []
        actual = self.df.groupby("category")["amount"].sum()
        for cat, budget in self.budgets.items():
            spent = actual.get(cat, 0)
            pct   = spent / budget * 100 if budget > 0 else 0
            if pct > 100:
                results.append(Insight(
                    f"Over Budget: {cat}",
                    f"Spent Rs {spent:,.0f} vs Rs {budget:,.0f} budget ({pct:.0f}% used). Cut back by Rs {spent-budget:,.0f}.",
                    "critical", "Budget"
                ))
            elif pct > 85:
                results.append(Insight(
                    f"Near Limit: {cat}",
                    f"At {pct:.0f}% of your Rs {budget:,.0f} budget. Only Rs {budget-spent:,.0f} remaining.",
                    "warning", "Budget"
                ))
            else:
                results.append(Insight(
                    f"On Track: {cat}",
                    f"Used {pct:.0f}% of Rs {budget:,.0f} budget. Rs {budget-spent:,.0f} available.",
                    "positive", "Budget"
                ))
        return results

    def _savings_rate_insight(self) -> List[Insight]:
        """Estimate savings if income is not present (compare spend to benchmark)."""
        monthly = self.df.groupby("month_dt")["amount"].sum()
        avg_monthly = monthly.mean()
        # Use Rs 50,000 as typical income benchmark for insights
        INCOME_ESTIMATE = 50_000
        savings_rate = max(0, (INCOME_ESTIMATE - avg_monthly) / INCOME_ESTIMATE * 100)
        if savings_rate < 10:
            return [Insight("Low Savings Rate", f"Estimated savings: only {savings_rate:.0f}% of income. Target 20%+ for financial health.", "critical", "Savings")]
        elif savings_rate < 20:
            return [Insight("Moderate Savings", f"Estimated savings: {savings_rate:.0f}%. Try to reach 20%+ by reducing discretionary spending.", "warning", "Savings")]
        return [Insight("Healthy Savings Rate", f"Estimated savings: {savings_rate:.0f}% — above the recommended 20% threshold.", "positive", "Savings")]

    def _subscription_creep_insight(self) -> List[Insight]:
        if "category" not in self.df.columns:
            return []
        monthly_cat = self.df.groupby(["month_dt", "category"])["amount"].sum().reset_index()
        creeping = []
        for cat, grp in monthly_cat.groupby("category"):
            if len(grp) >= 3:
                slope = np.polyfit(range(len(grp)), grp["amount"].values, 1)[0]
                if slope > 100:
                    creeping.append(f"{cat} (+Rs {slope:.0f}/month)")
        if creeping:
            return [Insight(
                "Subscription Creep Detected",
                f"These categories growing month-over-month: {', '.join(creeping)}. Review recurring charges.",
                "warning", "Pattern"
            )]
        return []

    def _category_diversity_insight(self) -> List[Insight]:
        n = self.df["category"].nunique()
        top3_share = self.df.groupby("category")["amount"].sum().nlargest(3).sum() / self.df["amount"].sum() * 100
        if top3_share > 80:
            return [Insight(
                "Highly Concentrated Spending",
                f"Top 3 categories = {top3_share:.0f}% of spend. Diversifying could improve financial resilience.",
                "info", "Structure"
            )]
        return []


def format_insight_for_cli(insight: Insight) -> str:
    """Format an Insight for terminal output."""
    sep = "─" * 60
    return f"\n{sep}\n{insight.icon} {insight.title} [{insight.category}]\n  {insight.body}\n"
