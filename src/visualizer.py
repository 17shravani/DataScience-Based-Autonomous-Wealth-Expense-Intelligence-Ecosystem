"""
src/visualizer.py — Phase 7: Visualization Module
===================================================
Generates and saves all charts to the outputs/ directory.
Uses Matplotlib + Seaborn.
Callable from both main.py (CLI) and the Streamlit dashboard.
"""

import os
import io
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.gridspec as gridspec
import seaborn as sns
from pathlib import Path
from typing import Optional, Tuple

# ── Global style ────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family"      : "DejaVu Sans",
    "axes.spines.top"  : False,
    "axes.spines.right": False,
    "axes.grid"        : True,
    "grid.alpha"       : 0.3,
    "grid.linestyle"   : "--",
})

PALETTE     = sns.color_palette("coolwarm", 10)
OUTPUT_DIR  = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

INR = mticker.FuncFormatter(lambda x, _: f"₹{x:,.0f}")


def _save(fig: plt.Figure, name: str, dpi: int = 140) -> str:
    """Save figure to outputs/ and return path."""
    path = OUTPUT_DIR / name
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


class ExpenseVisualizer:
    """
    All visualizations for the Expense Tracker.
    Each method returns a (fig, save_path) tuple.
    """

    def __init__(self, df: pd.DataFrame, theme: str = "light"):
        self.df    = df.copy()
        self.theme = theme
        self.bg    = "#FFFFFF" if theme == "light" else "#1C1C24"
        self.fg    = "#0F172A" if theme == "light" else "#F1F5F9"
        sns.set_theme(style="whitegrid" if theme == "light" else "darkgrid")

    def _base_fig(self, w=10, h=5):
        fig, ax = plt.subplots(figsize=(w, h), facecolor=self.bg)
        ax.set_facecolor(self.bg)
        for spine in ax.spines.values():
            spine.set_color("#CBD5E1")
        return fig, ax

    # ── Chart 1: Category Bar Chart ─────────────────────────────────────────────
    def category_bar(self) -> Tuple:
        cat = self.df.groupby("category")["amount"].sum().sort_values(ascending=True)
        fig, ax = self._base_fig(9, max(4, len(cat) * 0.55))
        colors = sns.color_palette("Blues_d", len(cat))
        bars = ax.barh(cat.index, cat.values, color=colors, height=0.65, edgecolor="white")
        ax.bar_label(bars, labels=[f"₹{v:,.0f}" for v in cat.values],
                     padding=6, fontsize=9, color=self.fg, fontweight="bold")
        ax.xaxis.set_major_formatter(INR)
        ax.set_title("💰 Category-Wise Total Spending", fontsize=14, fontweight="bold", color=self.fg, pad=12)
        ax.set_xlabel("Total Spent (₹)", color=self.fg, fontsize=10)
        ax.tick_params(colors=self.fg)
        fig.tight_layout()
        path = _save(fig, "1_category_bar.png")
        return fig, path

    # ── Chart 2: Category Pie Chart ─────────────────────────────────────────────
    def category_pie(self) -> Tuple:
        cat = self.df.groupby("category")["amount"].sum().sort_values(ascending=False)
        fig, ax = plt.subplots(figsize=(8, 6), facecolor=self.bg)
        ax.set_facecolor(self.bg)
        colors = sns.color_palette("Set2", len(cat))
        wedges, texts, autos = ax.pie(
            cat.values, labels=cat.index, autopct="%1.1f%%",
            colors=colors, startangle=140,
            wedgeprops=dict(edgecolor="white", linewidth=2),
            pctdistance=0.80,
        )
        for at in autos:
            at.set_fontsize(8.5); at.set_color("white"); at.set_fontweight("bold")
        ax.set_title("🥧 Spending Share by Category", fontsize=14, fontweight="bold", color=self.fg, pad=14)
        fig.tight_layout()
        path = _save(fig, "2_category_pie.png")
        return fig, path

    # ── Chart 3: Monthly Trend Line ─────────────────────────────────────────────
    def monthly_trend(self) -> Tuple:
        monthly = self.df.groupby("month_dt")["amount"].sum().reset_index().sort_values("month_dt")
        fig, ax = self._base_fig(11, 5)
        ax.plot(monthly["month_dt"], monthly["amount"],
                marker="o", linewidth=2.5, color="#2563EB",
                markersize=8, markerfacecolor="white", markeredgewidth=2.5)
        ax.fill_between(monthly["month_dt"], monthly["amount"], alpha=0.12, color="#2563EB")

        # Annotate each point
        for _, row in monthly.iterrows():
            ax.annotate(f"₹{row['amount']:,.0f}",
                        xy=(row["month_dt"], row["amount"]),
                        xytext=(0, 12), textcoords="offset points",
                        ha="center", fontsize=8, color="#2563EB", fontweight="bold")

        ax.yaxis.set_major_formatter(INR)
        ax.set_title("📅 Monthly Spending Trend", fontsize=14, fontweight="bold", color=self.fg, pad=12)
        ax.set_xlabel("Month", color=self.fg); ax.set_ylabel("Total Spend (₹)", color=self.fg)
        ax.tick_params(colors=self.fg); plt.xticks(rotation=35, ha="right")
        fig.tight_layout()
        path = _save(fig, "3_monthly_trend.png")
        return fig, path

    # ── Chart 4: Stacked Bar (Month × Category) ─────────────────────────────────
    def stacked_category_monthly(self) -> Tuple:
        pivot = self.df.pivot_table(index="month_dt", columns="category",
                                     values="amount", aggfunc="sum", fill_value=0)
        pivot.index = pivot.index.strftime("%b %Y") if hasattr(pivot.index[0], "strftime") else pivot.index
        fig, ax = self._base_fig(12, 5)
        pivot.plot(kind="bar", stacked=True, ax=ax,
                   color=sns.color_palette("tab10", len(pivot.columns)),
                   edgecolor="white", linewidth=0.5, width=0.65)
        ax.yaxis.set_major_formatter(INR)
        ax.set_title("📊 Monthly Spending by Category (Stacked)", fontsize=14, fontweight="bold", color=self.fg, pad=12)
        ax.set_xlabel(""); ax.set_ylabel("Amount (₹)", color=self.fg)
        ax.legend(loc="upper right", fontsize=8, ncols=2, framealpha=0.7)
        plt.xticks(rotation=35, ha="right"); ax.tick_params(colors=self.fg)
        fig.tight_layout()
        path = _save(fig, "4_stacked_monthly.png")
        return fig, path

    # ── Chart 5: Spending Heatmap ────────────────────────────────────────────────
    def heatmap(self) -> Tuple:
        pivot = self.df.pivot_table(index="category", columns="month",
                                     values="amount", aggfunc="sum", fill_value=0)
        fig, ax = plt.subplots(figsize=(max(8, len(pivot.columns) * 1.1), max(5, len(pivot) * 0.6)),
                                facecolor=self.bg)
        ax.set_facecolor(self.bg)
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlOrRd",
                    linewidths=0.5, linecolor=self.bg, ax=ax,
                    annot_kws={"size": 8, "weight": "bold"})
        ax.set_title("🔥 Spending Heatmap (Category × Month)", fontsize=14, fontweight="bold", color=self.fg, pad=12)
        ax.set_xlabel(""); ax.set_ylabel("")
        plt.xticks(rotation=40, ha="right"); plt.yticks(rotation=0)
        fig.tight_layout()
        path = _save(fig, "5_heatmap.png")
        return fig, path

    # ── Chart 6: Transaction Distribution Histogram ──────────────────────────────
    def amount_histogram(self) -> Tuple:
        fig, ax = self._base_fig(9, 4)
        sns.histplot(self.df["amount"], bins=30, kde=True, color="#7C3AED",
                     line_kws={"linewidth": 2.5}, ax=ax, alpha=0.75)
        ax.xaxis.set_major_formatter(INR)
        ax.set_title("📈 Transaction Amount Distribution", fontsize=14, fontweight="bold", color=self.fg, pad=12)
        ax.set_xlabel("Amount (₹)", color=self.fg); ax.set_ylabel("Frequency", color=self.fg)
        ax.tick_params(colors=self.fg)
        fig.tight_layout()
        path = _save(fig, "6_amount_distribution.png")
        return fig, path

    # ── Chart 7: Weekday Spending Pattern ───────────────────────────────────────
    def weekday_bar(self) -> Tuple:
        order = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
        day_avg = self.df.groupby("day_of_week")["amount"].mean().reindex(order).fillna(0)
        fig, ax = self._base_fig(9, 4)
        colors = ["#EF4444" if d in ["Saturday","Sunday"] else "#2563EB" for d in day_avg.index]
        bars = ax.bar(day_avg.index, day_avg.values, color=colors, edgecolor="white", width=0.65)
        ax.bar_label(bars, labels=[f"₹{v:,.0f}" for v in day_avg.values],
                     padding=4, fontsize=8, fontweight="bold", color=self.fg)
        ax.yaxis.set_major_formatter(INR)
        ax.set_title("📅 Avg Spend by Day of Week  (🔴 = Weekend)", fontsize=13, fontweight="bold", color=self.fg, pad=10)
        ax.set_xlabel(""); ax.set_ylabel("Avg Spend (₹)", color=self.fg)
        plt.xticks(rotation=20, ha="right"); ax.tick_params(colors=self.fg)
        fig.tight_layout()
        path = _save(fig, "7_weekday_pattern.png")
        return fig, path

    # ── Chart 8: Cumulative Spend Curve ─────────────────────────────────────────
    def cumulative_spend(self) -> Tuple:
        df = self.df.sort_values("date").copy()
        df["cumulative"] = df["amount"].cumsum()
        fig, ax = self._base_fig(11, 5)
        ax.plot(df["date"], df["cumulative"], color="#059669", linewidth=2.5)
        ax.fill_between(df["date"], df["cumulative"], alpha=0.12, color="#059669")
        ax.yaxis.set_major_formatter(INR)
        ax.set_title("📈 Cumulative Spending Over Time", fontsize=14, fontweight="bold", color=self.fg, pad=12)
        ax.set_xlabel("Date", color=self.fg); ax.set_ylabel("Cumulative Spend (₹)", color=self.fg)
        plt.xticks(rotation=35, ha="right"); ax.tick_params(colors=self.fg)
        fig.tight_layout()
        path = _save(fig, "8_cumulative_spend.png")
        return fig, path

    # ── Chart 9: Anomaly Scatter ─────────────────────────────────────────────────
    def anomaly_scatter(self) -> Tuple:
        df = self.df.copy()
        if "is_anomaly" not in df.columns:
            df["is_anomaly"] = False
        normal   = df[~df["is_anomaly"]]
        anomalies = df[df["is_anomaly"]]
        fig, ax = self._base_fig(11, 5)
        ax.scatter(normal["date"], normal["amount"], c="#3B82F6", alpha=0.5, s=25, label="Normal")
        ax.scatter(anomalies["date"], anomalies["amount"], c="#EF4444", s=80,
                   edgecolors="white", linewidths=1.5, zorder=5, label=f"Anomaly (n={len(anomalies)})")
        ax.yaxis.set_major_formatter(INR)
        ax.set_title("🔍 Transaction Anomaly Detection", fontsize=14, fontweight="bold", color=self.fg, pad=12)
        ax.set_xlabel("Date", color=self.fg); ax.set_ylabel("Amount (₹)", color=self.fg)
        ax.legend(fontsize=10); plt.xticks(rotation=35, ha="right"); ax.tick_params(colors=self.fg)
        fig.tight_layout()
        path = _save(fig, "9_anomaly_scatter.png")
        return fig, path

    # ── Chart 10: Summary Dashboard (4-panel) ────────────────────────────────────
    def summary_dashboard(self) -> Tuple:
        """A 2×2 summary chart for README / portfolio proof."""
        fig = plt.figure(figsize=(16, 12), facecolor=self.bg)
        gs  = gridspec.GridSpec(2, 2, figure=fig, hspace=0.45, wspace=0.35)

        # Panel 1 — Category Bar
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_facecolor(self.bg)
        cat = self.df.groupby("category")["amount"].sum().sort_values()
        colors = sns.color_palette("Blues_d", len(cat))
        ax1.barh(cat.index, cat.values, color=colors, edgecolor="white")
        ax1.xaxis.set_major_formatter(INR)
        ax1.set_title("Category Spend", fontweight="bold", color=self.fg, fontsize=12)
        ax1.tick_params(colors=self.fg)

        # Panel 2 — Monthly Trend
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_facecolor(self.bg)
        monthly = self.df.groupby("month_dt")["amount"].sum().sort_index()
        ax2.plot(monthly.index, monthly.values, marker="o", color="#2563EB", linewidth=2)
        ax2.fill_between(monthly.index, monthly.values, alpha=0.15, color="#2563EB")
        ax2.yaxis.set_major_formatter(INR)
        ax2.set_title("Monthly Trend", fontweight="bold", color=self.fg, fontsize=12)
        ax2.tick_params(colors=self.fg)
        plt.setp(ax2.xaxis.get_majorticklabels(), rotation=35, ha="right")

        # Panel 3 — Pie
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_facecolor(self.bg)
        cat2 = self.df.groupby("category")["amount"].sum()
        ax3.pie(cat2.values, labels=cat2.index, autopct="%1.0f%%",
                colors=sns.color_palette("Set2", len(cat2)), startangle=140,
                wedgeprops=dict(edgecolor="white", linewidth=1.5), pctdistance=0.80)
        ax3.set_title("Spend Share", fontweight="bold", color=self.fg, fontsize=12)

        # Panel 4 — Histogram
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_facecolor(self.bg)
        sns.histplot(self.df["amount"], bins=20, kde=True, color="#7C3AED",
                     ax=ax4, alpha=0.75, line_kws={"linewidth": 2})
        ax4.xaxis.set_major_formatter(INR)
        ax4.set_title("Amount Distribution", fontweight="bold", color=self.fg, fontsize=12)
        ax4.tick_params(colors=self.fg)

        fig.suptitle("📊 Expense Tracker — Summary Dashboard",
                      fontsize=18, fontweight="bold", color=self.fg, y=1.01)
        path = _save(fig, "0_summary_dashboard.png", dpi=120)
        return fig, path


# ── Tuple return type annotation fix ────────────────────────────────────────────
from typing import Tuple
