from __future__ import annotations

from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


try:
    from lightgbm import LGBMRegressor
except ImportError:  # pragma: no cover
    LGBMRegressor = None


# =============================================================================
# 1) Custom transformer (required by the project brief)
# =============================================================================
class DateFeaturesExtractor(BaseEstimator, TransformerMixin):
    """
    Extract simple calendar features from a datetime column.

    Notes
    -----
    - This is a simple custom transformer to satisfy the requirement of writing
      your own scikit-learn transformer (not Winsorizer / SquaredTransformer).
    - It outputs purely numeric features, so it can be treated as "numeric"
      downstream.

    Output columns
    --------------
    - month: 1..12
    - dayofweek: 0..6 (Mon=0)
    - is_weekend: 0/1
    """

    def __init__(self, date_col: str = "date"):
        self.date_col = date_col

    def fit(self, X, y=None):  # noqa: D401
        """No fitting required; returns self."""
        return self

    def transform(self, X):
        """
        Parameters
        ----------
        X : array-like or DataFrame
            Must contain the date column.

        Returns
        -------
        pd.DataFrame
            DataFrame with engineered date features.
        """
        X_df = pd.DataFrame(X).copy()

        if self.date_col not in X_df.columns:
            raise KeyError(f"Missing date column: {self.date_col}")

        dt = pd.to_datetime(X_df[self.date_col], errors="raise")

        out = pd.DataFrame(
            {
                "month": dt.dt.month.astype("int64"),
                "dayofweek": dt.dt.dayofweek.astype("int64"),
                "is_weekend": (dt.dt.dayofweek >= 5).astype("int64"),
            }
        )
        return out


# =============================================================================
# 2) Preprocessor builder (sklearn transformers inside a ColumnTransformer)
# =============================================================================
def build_preprocessor(
    date_col: str = "date",
    categorical_cols: Iterable[str] | None = None,
) -> ColumnTransformer:
    """
    Build a preprocessing step for mixed-type tabular data.

    Strategy
    --------
    - Numeric columns: median imputation
    - Categorical columns: most-frequent imputation + one-hot encoding
    - Date column: custom DateFeaturesExtractor -> numeric features

    Parameters
    ----------
    date_col : str
        Name of datetime column.
    categorical_cols : Iterable[str] | None
        Optionally specify categorical columns explicitly. If None, we use
        dtype-based selection for ['category','object','bool'].

    Returns
    -------
    ColumnTransformer
        A transformer suitable as the first stage in a modelling Pipeline.
    """
    # Selectors
    numeric_selector = make_column_selector(dtype_include=np.number)

    if categorical_cols is None:
        categorical_selector = make_column_selector(
            dtype_include=["category", "object", "bool"]
        )
    else:
        categorical_selector = list(categorical_cols)

    # Numeric pipeline
    num_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
        ]
    )

    # Categorical pipeline
    cat_pipe = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    # Date pipeline (custom transformer)
    date_pipe = Pipeline(
        steps=[
            ("date_feats", DateFeaturesExtractor(date_col=date_col)),
            ("imputer", SimpleImputer(strategy="most_frequent")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_pipe, numeric_selector),
            ("cat", cat_pipe, categorical_selector),
            ("date", date_pipe, [date_col]),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    return preprocessor


# =============================================================================
# 3) Pipeline builders (GLM-ish + LGBM)
# =============================================================================
def build_glm_pipeline(
    date_col: str = "date",
    categorical_cols: Iterable[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a GLM-style pipeline.

    Why ElasticNet?
    ---------------
    The project brief requires tuning 'alpha' and 'l1_ratio' for the GLM pipeline.
    scikit-learn's ElasticNet exposes exactly these hyperparameters and integrates
    cleanly in a Pipeline.

    Returns
    -------
    Pipeline
        preprocess -> ElasticNet
    """
    pre = build_preprocessor(date_col=date_col, categorical_cols=categorical_cols)

    model = ElasticNet(
        alpha=1.0,       # tuned later
        l1_ratio=0.5,    # tuned later
        max_iter=5000,
        random_state=random_state,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )
    return pipe


def build_lgbm_pipeline(
    date_col: str = "date",
    categorical_cols: Iterable[str] | None = None,
    random_state: int = 42,
) -> Pipeline:
    """
    Build an LGBM pipeline.

    Returns
    -------
    Pipeline
        preprocess -> LightGBM regressor

    Raises
    ------
    ImportError
        If lightgbm is not installed.
    """
    if LGBMRegressor is None:
        raise ImportError("lightgbm is required for build_lgbm_pipeline().")

    pre = build_preprocessor(date_col=date_col, categorical_cols=categorical_cols)

    model = LGBMRegressor(
        objective="regression",
        random_state=random_state,
    )

    pipe = Pipeline(
        steps=[
            ("preprocess", pre),
            ("model", model),
        ]
    )
    return pipe
