from __future__ import annotations

import pandas as pd


def _snake_case_columns(cols: pd.Index) -> pd.Index:
    """Convert column names to snake_case and remove special characters."""
    return (
        cols.astype(str)
        .str.strip()
        .str.lower()
        .str.replace(r"\s+", "_", regex=True)
        .str.replace("(", "", regex=False)
        .str.replace(")", "", regex=False)
        .str.replace("°", "", regex=False)
        .str.replace("/", "_", regex=False)
    )


def clean_seoul_bike_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Seoul Bike Sharing Demand dataset for modelling.

    EDA-driven decisions:
    - The dataset contains no missing values => no imputation required.
    - Extreme values in weather variables (rainfall/snowfall) are retained as they
      reflect genuine conditions rather than data errors.
    - Date is parsed to datetime for downstream temporal feature extraction.
    - Categorical variables are cast to 'category' dtype for modelling pipelines.

    Parameters
    ----------
    df : pd.DataFrame
        Raw dataset loaded from CSV.

    Returns
    -------
    pd.DataFrame
        Cleaned dataset with consistent column names and dtypes.
    """
    out = df.copy()

    # 1) Standardise column names
    out.columns = _snake_case_columns(out.columns)

    # 2) Parse date
    # raw column is "Date" -> after renaming becomes "date"
    if "date" not in out.columns:
        raise KeyError(
            "Expected a 'date' column after renaming. "
            "Please check the raw dataset columns."
        )
    out["date"] = pd.to_datetime(out["date"], format="%d/%m/%Y", errors="raise")

    # 3) Cast categorical variables
    cat_cols = ["seasons", "holiday", "functioning_day"]
    missing_cats = [c for c in cat_cols if c not in out.columns]
    if missing_cats:
        raise KeyError(
            f"Missing expected categorical columns: {missing_cats}. "
            "Please check the raw dataset columns."
        )
    for c in cat_cols:
        out[c] = out[c].astype("category")

    # 4) Light sanity checks (no aggressive cleaning)
    # target is "rented_bike_count" after renaming
    if "rented_bike_count" not in out.columns:
        raise KeyError(
            "Expected 'rented_bike_count' column after renaming. "
            "Please check the raw dataset columns."
        )

    # Optional: ensure non-negative target (dataset definition implies counts)
    if (out["rented_bike_count"] < 0).any():
        raise ValueError("Found negative values in 'rented_bike_count'.")

    # Keep original numeric dtypes; no forced casting needed.

    return out
