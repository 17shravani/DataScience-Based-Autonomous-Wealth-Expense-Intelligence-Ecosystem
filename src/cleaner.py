"""
src/cleaner.py — Phase 3: Data Cleaning Module
================================================
Handles all cleaning and standardization of raw expense DataFrames:
  - Column renaming & type coercion
  - Date parsing
  - Amount normalization (always positive)
  - Missing value imputation
  - Duplicate removal
  - Derived time columns (Month, Week, Quarter, DayOfWeek)
  - Spending tier labeling
"""

import pandas as pd
import numpy as np


# ── Constants ────────────────────────────────────────────────────────────────────
REQUIRED_COLS = {"date", "category", "amount"}

CATEGORY_ALIASES = {
    "food":          "Food & Dining",
    "dining":        "Food & Dining",
    "restaurant":    "Food & Dining",
    "grocery":       "Groceries",
    "groceries":     "Groceries",
    "transport":     "Transport",
    "transportation":"Transport",
    "travel":        "Travel",
    "rent":          "Housing",
    "housing":       "Housing",
    "utilities":     "Utilities",
    "utility":       "Utilities",
    "shopping":      "Shopping",
    "entertainment": "Entertainment",
    "healthcare":    "Healthcare",
    "health":        "Healthcare",
    "education":     "Education",
    "subscriptions": "Subscriptions",
    "subscription":  "Subscriptions",
}

SPENDING_TIERS = [
    (0,    500,   "Low"),
    (500,  2000,  "Medium"),
    (2000, 10000, "High"),
    (10000, np.inf, "Very High"),
]


# ── Main Function ────────────────────────────────────────────────────────────────
def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline for an expense DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        Raw input (from CSV upload, manual entry, or synthetic data).

    Returns
    -------
    pd.DataFrame
        A clean, enriched DataFrame ready for analysis.

    Raises
    ------
    ValueError
        If required columns are missing after normalisation.
    """
    df = df.copy()

    # ── Step 1: Normalize column names ─────────────────────────────────────────
    df.columns = df.columns.str.strip().str.lower().str.replace(r"\s+", "_", regex=True)

    # ── Step 2: Validate required columns ──────────────────────────────────────
    missing = REQUIRED_COLS - set(df.columns)
    if missing:
        raise ValueError(
            f"Missing required columns: {missing}. "
            f"Your CSV must have: Date, Category, Amount"
        )

    # ── Step 3: Parse dates ─────────────────────────────────────────────────────
    df["date"] = pd.to_datetime(df["date"], dayfirst=False, errors="coerce")
    invalid_dates = df["date"].isna().sum()
    if invalid_dates > 0:
        print(f"  [Cleaner] [WARN] Dropped {invalid_dates} rows with unparseable dates.")
    df = df.dropna(subset=["date"])

    # ── Step 4: Clean amounts ───────────────────────────────────────────────────
    df["amount"] = pd.to_numeric(df["amount"], errors="coerce").abs()
    invalid_amounts = df["amount"].isna().sum()
    if invalid_amounts > 0:
        print(f"  [Cleaner] [WARN] Dropped {invalid_amounts} rows with non-numeric amounts.")
    df = df.dropna(subset=["amount"])
    df = df[df["amount"] > 0]  # remove zero-amount rows

    # ── Step 5: Standardize categories ─────────────────────────────────────────
    df["category"] = (
        df["category"]
        .fillna("Other")
        .str.strip()
        .str.lower()
        .map(lambda c: CATEGORY_ALIASES.get(c, c.title()))
    )

    # ── Step 6: Optional columns ────────────────────────────────────────────────
    if "description" not in df.columns:
        df["description"] = "—"
    else:
        df["description"] = df["description"].fillna("—").str.strip()

    if "payment_method" not in df.columns:
        df["payment_method"] = "Unknown"
    else:
        df["payment_method"] = df["payment_method"].fillna("Unknown").str.strip().str.title()

    # ── Step 7: Remove duplicates ───────────────────────────────────────────────
    before = len(df)
    df = df.drop_duplicates(subset=["date", "amount", "category"])
    dupes = before - len(df)
    if dupes > 0:
        print(f"  [Cleaner] [INFO] Removed {dupes} duplicate rows.")

    # ── Step 8: Derived time columns (Phase 5 Feature Engineering) ─────────────
    df["month"]       = df["date"].dt.to_period("M").astype(str)
    df["month_dt"]    = df["date"].dt.to_period("M").dt.to_timestamp()
    df["week"]        = df["date"].dt.isocalendar().week.astype(int)
    df["year"]        = df["date"].dt.year
    df["quarter"]     = df["date"].dt.quarter.map({1:"Q1", 2:"Q2", 3:"Q3", 4:"Q4"})
    df["day_of_week"] = df["date"].dt.day_name()
    df["is_weekend"]  = df["date"].dt.dayofweek >= 5

    # ── Step 9: Spending tier labels ────────────────────────────────────────────
    def _tier(amount):
        for lo, hi, label in SPENDING_TIERS:
            if lo <= amount < hi:
                return label
        return "Very High"

    df["spending_tier"] = df["amount"].apply(_tier)

    # ── Step 10: Sort and reset index ───────────────────────────────────────────
    df = df.sort_values("date").reset_index(drop=True)

    print(f"  [Cleaner] [OK] Clean dataset: {len(df)} rows, {df['category'].nunique()} categories.")
    return df
