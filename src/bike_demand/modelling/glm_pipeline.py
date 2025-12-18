from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, Optional, Tuple

import pandas as pd
from glum import GeneralizedLinearRegressor, TweedieDistribution
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from bike_demand.feature_engineering.transformers import CyclicalEncoder


# -------------------------
# Feature specification
# -------------------------
@dataclass(frozen=True)
class BikeFeatureSpec:
    """
    Single source of truth for feature groups.

    Note:
    - Pipelines should not crash if some columns are missing (e.g. older parquet).
      We handle this by filtering to columns that exist at fit-time.
    """

    target: str = "rented_bike_count"

    # cyclical variables (will be encoded into sin/cos and originals dropped)
    cyclical: Tuple[str, ...] = ("hour", "month", "day_of_week")

    # continuous numeric predictors (edit this list as your cleaned data supports)
    numeric: Tuple[str, ...] = (
        "dew_point_temp",
        "temperature",
        "humidity",
        "wind_speed",
        "visibility",
        "solar_radiation"
    )

    # categorical predictors
    categorical: Tuple[str, ...] = ()

    # optional binary flags (treat as categorical for GLM)
    binary: Tuple[str, ...] = ("rainfall_binary", "snowfall_binary","holiday")

    # always drop from X
    drop: Tuple[str, ...] = (
    "date",
    "sample",
    "functioning_day",
    "seasons",
    "rainfall",
    "snowfall"
    )

def split_xy(
    df: pd.DataFrame,
    spec: BikeFeatureSpec | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Split dataframe into X / y. Drops target and spec.drop from X (if present).
    """
    spec = spec or BikeFeatureSpec()
    X = df.drop(columns=[spec.target, *spec.drop], errors="ignore")
    y = df[spec.target]
    return X, y


# -------------------------
# Pipeline
# -------------------------
class _ColumnAwarePreprocess(BaseEstimator, TransformerMixin):
    """
    A lightweight wrapper that builds the ColumnTransformer at fit-time
    based on the columns that actually exist in X.
    """

    def __init__(self, spec: BikeFeatureSpec):
        self.spec = spec

    def fit(self, X: pd.DataFrame, y=None):
        cols = set(X.columns)

        # base numeric/categorical features that exist
        numeric_base = [c for c in self.spec.numeric if c in cols]
        categorical_base = [c for c in self.spec.categorical if c in cols]
        binary_base = [c for c in self.spec.binary if c in cols]

        # cyclical sin/cos columns (created by CyclicalEncoder if original existed)
        cyc_num: list[str] = []
        for c in self.spec.cyclical:
            s, co = f"{c}_sin", f"{c}_cos"
            if s in cols and co in cols:
                cyc_num += [s, co]

        numeric_features = numeric_base + cyc_num
        categorical_features = categorical_base + binary_base

        num_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]
        )

        self.pre_ = ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_features),
                ("cat", cat_pipe, categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

        self.pre_.fit(X, y)
        return self

    def transform(self, X: pd.DataFrame):
        return self.pre_.transform(X)


def build_glm_pipeline(
    spec: Optional[BikeFeatureSpec] = None,
    *,
    tweedie_power: float = 1.5,
) -> Pipeline:
    """
    GLM pipeline:
    1) Cyclical encode hour/month/day_of_week -> sin/cos (drop originals)
    2) Impute + scale numeric
    3) Impute + one-hot categorical/binary
    4) GeneralizedLinearRegressor with Tweedie distribution

    alpha and l1_ratio can be tuned via GridSearchCV:
      - model__alpha
      - model__l1_ratio
    """
    spec = spec or BikeFeatureSpec()

    cyc = CyclicalEncoder(
        columns=list(spec.cyclical),
        periods={"hour": 24, "month": 12, "day_of_week": 7},
        drop_original=True,
    )

    model = GeneralizedLinearRegressor(
        family=TweedieDistribution(tweedie_power),
        fit_intercept=True,
        alpha=0.0,      # tune
        l1_ratio=0.0,   # tune
    )

    return Pipeline(
        steps=[
            ("cyclical", cyc),
            ("preprocess", _ColumnAwarePreprocess(spec)),
            ("model", model),
        ]
    )
