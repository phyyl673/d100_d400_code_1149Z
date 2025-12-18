from __future__ import annotations
import numpy as np
from pathlib import Path
import pandas as pd


def clean_seoul_bike_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean and prepare the Seoul Bike Sharing Demand dataset for modelling.

    Changes applied:
    - Stripped column names and renamed columns using an explicit mapping.
    - Parsed dates and extracted calendar features (month, day_of_week, is_weekend).
    - Standardised categorical variables (strip values) and cast to 'category' dtype.
    - Replaced +/-inf with NaN and performed basic sanity checks.
    - Retained outliers as they reflect genuine weather conditions.
    """
    out = df.copy()
    out.columns = out.columns.str.strip()

    rename_map = {
        "Date": "date",
        "Rented Bike Count": "rented_bike_count",
        "Hour": "hour",
        "Temperature(°C)": "temperature",
        "Humidity(%)": "humidity",
        "Wind speed (m/s)": "wind_speed",
        "Visibility (10m)": "visibility",
        "Dew point temperature(°C)": "dew_point_temp",
        "Solar Radiation (MJ/m2)": "solar_radiation",
        "Rainfall(mm)": "rainfall",
        "Snowfall (cm)": "snowfall",
        "Seasons": "seasons",
        "Holiday": "holiday",
        "Functioning Day": "functioning_day",
    }

    required = set(rename_map.keys())
    missing = required - set(out.columns)
    if missing:
        raise KeyError(f"Raw dataset missing expected columns: {sorted(missing)}")

    out = out.rename(columns=rename_map)

    # Parse date (dataset is dd/mm/yyyy)
    out["date"] = pd.to_datetime(out["date"], format="%d/%m/%Y", errors="raise")

    # Ensure hour is integer and within expected range
    out["hour"] = out["hour"].astype("int16")
    if (out["hour"] < 0).any() or (out["hour"] > 23).any():
        raise ValueError("Found 'hour' values outside expected range [0, 23].")

    # Calendar features
    out["month"] = out["date"].dt.month.astype("int8")
    out["day_of_week"] = out["date"].dt.dayofweek.astype("int8")  # Monday=0
    
    # Standardise + cast categoricals
    cat_cols = ["seasons", "holiday", "functioning_day"]
    for c in cat_cols:
        out[c] = out[c].astype(str).str.strip().astype("category")

    # Replace infinite numeric values with NaN 
    num_cols = [
        "rented_bike_count",
        "temperature",
        "humidity",
        "wind_speed",
        "visibility",
        "dew_point_temp",
        "solar_radiation",
        "rainfall",
        "snowfall",
    ]
    out[num_cols] = out[num_cols].replace([np.inf, -np.inf], np.nan)

    # Sanity check for target
    if (out["rented_bike_count"] < 0).any():
        raise ValueError("Found negative values in 'rented_bike_count'.")

    # If NaNs appear after cleaning (unexpected for this dataset), fail fast
    if out[num_cols].isna().any().any():
        nan_cols = out[num_cols].columns[out[num_cols].isna().any()].tolist()
        raise ValueError(f"Found NaN values after cleaning in columns: {nan_cols}")

    return out


def find_project_root(start: Path) -> Path:
    for p in [start] + list(start.parents):
        if (p / "pyproject.toml").exists():
            return p
    raise FileNotFoundError("Could not find project root containing pyproject.toml")

def save_processed_data(
    df: pd.DataFrame,
    filename: str = "seoul_bike_cleaned.parquet",
) -> Path:
    project_root = find_project_root(Path(__file__).resolve())
    processed_dir = project_root / "data" / "processed"
    processed_dir.mkdir(parents=True, exist_ok=True)

    path = processed_dir / filename
    print("Saving columns:", df.columns.tolist())
    print("Saving to:", path.resolve())
    df.to_parquet(path, index=False)
    
    return path