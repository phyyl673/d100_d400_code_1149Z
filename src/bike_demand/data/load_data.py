from pathlib import Path
import pandas as pd


def load_data(csv_path: str = "raw/SeoulBikeData.csv") -> pd.DataFrame:
    """
    Load the project's dataset.

    The dataset is assumed to be located under the project's
    `data/` directory. The default relative path can
    be changed if the data location is updated.

    Parameters
    ----------
    csv_path : str, optional
        Relative path to the data file within the project's
        `data/` directory.
        Defaults to "raw/SeoulBikeData.csv".

    Returns
    -------
    pd.DataFrame
        Dataset loaded as a pandas DataFrame.
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

def load_cleaned_data(
    parquet_path: str = "processed/seoul_bike_cleaned.parquet",
    engine: str | None = "fastparquet",
) -> pd.DataFrame:
    """
    Load the cleaned dataset saved as parquet.

    Parameters
    ----------
    parquet_path : str
        Relative path within the data directory.
    engine : str | None
        Parquet engine to use ("fastparquet" recommended for compatibility).

    Returns
    -------
    pd.DataFrame
        Cleaned dataset loaded as a DataFrame.
    """
    project_root = Path(__file__).resolve().parents[3]
    data_dir = project_root / "data"

    file_path = data_dir / parquet_path

    if not file_path.exists():
        raise FileNotFoundError(
            f"Cleaned parquet file not found at expected location: {file_path}"
        )

    return pd.read_parquet(file_path, engine=engine)