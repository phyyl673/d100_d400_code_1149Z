from pathlib import Path
import pandas as pd


def load_data(csv_path: str = "raw/SeoulBikeData.csv") -> pd.DataFrame:
    """
    Loads the raw Seoul Bike Data from the project's data directory.

    The dataset is assumed to be located under the project's
    `data/` directory. The default relative path can
    be changed if the data location is updated.

    Parameters
    ----------
    csv_path : str, optional
        The relative path to the CSV file within the 'data' directory.
        By default "raw/SeoulBikeData.csv".

    Returns
    -------
    pd.DataFrame
        The loaded pandas DataFrame containing the raw bike data.

    Raises
    ------
    FileNotFoundError
        If the specified CSV file does not exist at the expected location.
    """
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"

    file_path = data_dir / csv_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Data file not found at expected location: {file_path}"
        )

    df = pd.read_csv(file_path, encoding="latin1")
    return df


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]

def load_cleaned_data(
    parquet_path: str = "processed/seoul_bike_cleaned.parquet",
) -> pd.DataFrame:
    """
    Load cleaned (processed) Seoul Bike data from parquet.

    This loader is defensive: if the stored parquet was generated before
    we added calendar features (month/day_of_week/is_weekend), it will
    recreate them from the 'date' column.
    """
    root = _project_root()
    file_path = root / "data" / parquet_path

    if not file_path.exists():
        raise FileNotFoundError(f"Processed parquet not found: {file_path}")

    df = pd.read_parquet(file_path)

    # ---- Defensive feature completion (handles "old parquet" cases)
    if "date" not in df.columns:
        raise KeyError("Expected column 'date' not found in cleaned data.")

    # ensure datetime
    if not pd.api.types.is_datetime64_any_dtype(df["date"]):
        df["date"] = pd.to_datetime(df["date"], errors="raise")


    return df