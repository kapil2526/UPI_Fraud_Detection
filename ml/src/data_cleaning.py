"""
data_cleaning.py
----------------
Cleans the raw UPI transaction data and saves data/processed/cleaned_data.csv

CLEANING DECISIONS (also saved to reports/cleaning_summary.txt):
1. Column names renamed to snake_case (no spaces, no brackets).
2. Missing values: numbers -> median, text -> most common value (mode).
   (The raw file has no missing values, but the function is reusable.)
3. Duplicates: remove repeated transaction ids and fully repeated rows.
4. Data types: timestamp -> datetime, text columns -> stripped strings.
5. Invalid values: amount must be > 0 and <= 100000 (UPI limit),
   fraud_flag must be 0 or 1, hour must be 0-23.
6. Outliers: only REPORTED, not removed. Big amounts can be real fraud.
7. No columns are dropped here (team must decide first).
"""

import os
import pandas as pd

# ---------- File paths ----------
# This finds the main project folder, so the code works from anywhere
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
RAW_PATH = os.path.join(BASE_DIR, "data", "raw", "upi_transactions_2024_raw.csv")
CLEAN_PATH = os.path.join(BASE_DIR, "data", "processed", "cleaned_data.csv")
SUMMARY_PATH = os.path.join(BASE_DIR, "reports", "cleaning_summary.txt")

# Old column name -> new column name
NEW_NAMES = {
    "transaction id": "transaction_id",
    "transaction type": "transaction_type",
    "amount (INR)": "amount_inr",
}


def load_data(path=RAW_PATH):
    """Read the csv file into a DataFrame."""
    df = pd.read_csv(path)
    return df


def rename_columns(df):
    """Make column names simple and consistent."""
    df = df.rename(columns=NEW_NAMES)
    df.columns = df.columns.str.strip().str.lower()
    return df


def quality_summary(df):
    """Return a small dictionary describing the quality of the data."""
    summary = {
        "rows": df.shape[0],
        "columns": df.shape[1],
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
    }
    if "transaction_id" in df.columns:
        summary["duplicate_ids"] = int(df["transaction_id"].duplicated().sum())
    if "fraud_flag" in df.columns:
        summary["fraud_count"] = int((df["fraud_flag"] == 1).sum())
    return summary


def handle_missing_values(df):
    """Fill missing values. Numbers get the median, text gets the mode."""
    for col in df.columns:
        if df[col].isnull().sum() == 0:
            continue  # nothing to fix in this column
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode()[0])
    # A row with no target label is useless for training, so drop it
    df = df.dropna(subset=["fraud_flag"])
    return df


def remove_duplicates(df):
    """Remove repeated transaction ids and repeated rows."""
    df = df.drop_duplicates(subset="transaction_id", keep="first")
    # Same transaction but with a different id is also suspicious
    other_cols = [c for c in df.columns if c != "transaction_id"]
    df = df.drop_duplicates(subset=other_cols, keep="first")
    return df


def fix_data_types(df):
    """Make sure every column has the correct type."""
    df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    df = df.dropna(subset=["timestamp"])  # drop rows with unreadable dates

    # Remove extra spaces in text columns (columns that are not numbers or dates)
    for col in df.columns:
        is_number = pd.api.types.is_numeric_dtype(df[col])
        is_date = pd.api.types.is_datetime64_any_dtype(df[col])
        if not is_number and not is_date:
            df[col] = df[col].astype(str).str.strip()

    df["amount_inr"] = pd.to_numeric(df["amount_inr"], errors="coerce")
    df["fraud_flag"] = df["fraud_flag"].astype(int)
    df["hour_of_day"] = df["hour_of_day"].astype(int)
    df["is_weekend"] = df["is_weekend"].astype(int)
    return df


def remove_invalid_values(df):
    """Remove rows that are impossible or wrong."""
    # Amount must be positive and under the UPI limit of 1 lakh
    df = df[(df["amount_inr"] > 0) & (df["amount_inr"] <= 100000)]
    # Target must be 0 or 1
    df = df[df["fraud_flag"].isin([0, 1])]
    # Hour must be 0 to 23
    df = df[(df["hour_of_day"] >= 0) & (df["hour_of_day"] <= 23)]
    # is_weekend must be 0 or 1
    df = df[df["is_weekend"].isin([0, 1])]
    return df


def fix_time_columns(df):
    """Make hour, day and weekend columns match the timestamp."""
    df["hour_of_day"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.day_name()
    df["is_weekend"] = (df["timestamp"].dt.dayofweek >= 5).astype(int)
    return df


def report_outliers(df, column="amount_inr"):
    """Count outliers with the IQR rule. We only REPORT them, not remove."""
    q1 = df[column].quantile(0.25)
    q3 = df[column].quantile(0.75)
    iqr = q3 - q1
    upper = q3 + 1.5 * iqr
    lower = q1 - 1.5 * iqr
    n_outliers = int(((df[column] > upper) | (df[column] < lower)).sum())
    return {"upper_limit": upper, "lower_limit": lower, "outlier_count": n_outliers}


def clean_data(df):
    """Run all cleaning steps in order."""
    df = rename_columns(df)
    df = handle_missing_values(df)
    df = remove_duplicates(df)
    df = fix_data_types(df)
    df = remove_invalid_values(df)
    df = fix_time_columns(df)
    df = df.reset_index(drop=True)
    return df


def main():
    raw = load_data()
    before = quality_summary(rename_columns(raw.copy()))

    cleaned = clean_data(raw.copy())
    after = quality_summary(cleaned)
    outliers = report_outliers(cleaned)

    # Save the cleaned data
    os.makedirs(os.path.dirname(CLEAN_PATH), exist_ok=True)
    cleaned.to_csv(CLEAN_PATH, index=False)

    # Build a before/after text report
    lines = ["BEFORE vs AFTER CLEANING", "=" * 30]
    for key in before:
        lines.append(f"{key:18s} before: {before[key]:>8}   after: {after.get(key, '-'):>8}")
    lines.append("")
    lines.append(f"Outliers in amount_inr (IQR rule): {outliers['outlier_count']} (kept, not removed)")
    lines.append(f"Outlier upper limit: {outliers['upper_limit']:.1f}")
    lines.append(f"Rows removed in total: {before['rows'] - after['rows']}")
    report = "\n".join(lines)

    os.makedirs(os.path.dirname(SUMMARY_PATH), exist_ok=True)
    with open(SUMMARY_PATH, "w") as f:
        f.write(report)

    print(report)
    print("\nSaved:", CLEAN_PATH)


if __name__ == "__main__":
    main()