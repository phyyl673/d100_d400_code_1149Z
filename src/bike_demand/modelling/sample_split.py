from __future__ import annotations

import hashlib
from typing import Iterable, Literal, Optional

import pandas as pd


SplitMethod = Literal["random", "id_hash"]


def create_sample_split_random(
    df: pd.DataFrame,
    *,
    train_frac: float = 0.8,
    sample_col: str = "sample",
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Random train/validation split that creates a `sample` column.

    Notes
    -----
    - Appropriate under i.i.d. assumptions.
    - Reproducible within the same environment via `random_state`,

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    train_frac : float, optional
        Fraction assigned to the training set. By default 0.8.
    sample_col : str, optional
        Name of the output split column. By default "sample".
    random_state : int, optional
        Seed for reproducibility. By default 42.

    Returns
    -------
    pd.DataFrame
        Copy of df with `sample` column taking values
        {"train", "validation"}.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")

    out = df.copy()

    train_idx = out.sample(frac=train_frac, random_state=random_state).index
    out[sample_col] = "validation"
    out.loc[train_idx, sample_col] = "train"

    return out


def create_sample_split_id_hash(
    df: pd.DataFrame,
    *,
    key_cols: Iterable[str],
    train_frac: float = 0.8,
    sample_col: str = "sample",
    modulo: int = 100,
) -> pd.DataFrame:
    """
    Deterministic ID-based train/validation split using a stable hash
    of key columns.

    In the seoul bike sharing demand forecasting project, there is no 
    explicit ID column.A composite keyn(e.g. ['date', 'hour']) can be 
    used instead.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    key_cols : Iterable[str]
        Column(s) used to construct a unique key per observation
        (e.g. ['date', 'hour']).
    train_frac : float, optional
        Fraction assigned to the training set. By default 0.8.
    sample_col : str, optional
        Name of the output split column. By default "sample".
    modulo : int, optional
        Number of buckets used for hashing. By default 100.

    Returns
    -------
    pd.DataFrame
        Copy of df with `sample` column taking values
        {"train", "validation"}.

    Raises
    ------
    KeyError
        If any of the key columns are missing.
    """
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must be between 0 and 1.")
    if modulo <= 1:
        raise ValueError("modulo must be > 1.")

    key_cols = list(key_cols)
    missing = set(key_cols) - set(df.columns)
    if missing:
        raise KeyError(f"Key column(s) not found in DataFrame: {missing}")

    out = df.copy()
    threshold = int(train_frac * modulo)

    def _bucket_for_row(row: pd.Series) -> int:
        key = "_".join(str(row[col]) for col in key_cols)
        h = hashlib.md5(key.encode("utf-8")).hexdigest()
        return int(h, 16) % modulo

    buckets = out.apply(_bucket_for_row, axis=1)
    out[sample_col] = buckets.map(
        lambda b: "train" if b < threshold else "validation"
    )

    return out


def create_sample_split(
    df: pd.DataFrame,
    *,
    method: SplitMethod,
    train_frac: float = 0.8,
    sample_col: str = "sample",
    key_cols: Optional[Iterable[str]] = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Convenience wrapper for creating a train/validation split.

    Examples
    --------
    >>> df = create_sample_split(
    ...     df, method="id_hash", key_cols=["date", "hour"]
    ... )
    >>> df = create_sample_split(
    ...     df, method="random", random_state=42
    ... )
    """
    if method == "random":
        return create_sample_split_random(
            df,
            train_frac=train_frac,
            sample_col=sample_col,
            random_state=random_state,
        )

    if method == "id_hash":
        if key_cols is None:
            raise ValueError("key_cols must be provided when method='id_hash'.")
        return create_sample_split_id_hash(
            df,
            key_cols=key_cols,
            train_frac=train_frac,
            sample_col=sample_col,
        )

    raise ValueError(f"Unknown split method: {method}")