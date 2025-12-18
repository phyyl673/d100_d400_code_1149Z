from __future__ import annotations

from typing import Optional

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bike_demand.feature_engineering.transformers import CyclicalEncoder
from bike_demand.modelling.glm_pipeline import BikeFeatureSpec


def split_xy(
    df: pd.DataFrame, spec: BikeFeatureSpec | None = None
) -> tuple[pd.DataFrame, pd.Series]:
    spec = spec or BikeFeatureSpec()
    X = df.drop(columns=[spec.target, *spec.drop], errors="ignore")
    y = df[spec.target]
    return X, y


def build_lgbm_pipeline(
    spec: Optional[BikeFeatureSpec] = None,
    *,
    tweedie_power: float = 1.5,
    random_state: int = 42,
) -> Pipeline:
    """
    Build a LightGBM pipeline with model-specific preprocessing.

    Notes
    -----
    - Cyclical variables are expanded using sine/cosine encoding.
    - Binary features are split into:
        * numeric binary (e.g. 0/1) -> treated as numeric
        * string/label binary (e.g. "Holiday"/"No Holiday") -> treated as categorical
      This prevents numeric imputers (median) from failing on non-numeric binaries.
    """
    spec = spec or BikeFeatureSpec()

    cyc = CyclicalEncoder(
        columns=list(spec.cyclical),
        periods={"hour": 24, "month": 12, "day_of_week": 7},
        drop_original=True,
    )

    def _make_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
        cols = set(X.columns)

        numeric_base = [c for c in spec.numeric if c in cols]
        categorical_base = [c for c in spec.categorical if c in cols]

        # --- split binary into numeric vs categorical by dtype ---
        binary_numeric: list[str] = []
        binary_categorical: list[str] = []
        for c in getattr(spec, "binary", []):
            if c in cols:
                if pd.api.types.is_numeric_dtype(X[c]):
                    binary_numeric.append(c)
                else:
                    binary_categorical.append(c)

        # cyclical features produced by CyclicalEncoder
        cyc_num: list[str] = []
        for c in spec.cyclical:
            sin_col, cos_col = f"{c}_sin", f"{c}_cos"
            if sin_col in cols and cos_col in cols:
                cyc_num.extend([sin_col, cos_col])

        numeric_features = numeric_base + binary_numeric + cyc_num
        categorical_features = categorical_base + binary_categorical

        num_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="median")),
            ]
        )

        cat_pipe = Pipeline(
            steps=[
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_features),
                ("cat", cat_pipe, categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    class _ColumnAwarePreprocess(BaseEstimator, TransformerMixin):
        """
        Column-aware wrapper that builds the ColumnTransformer based on
        the columns observed at fit time.
        """

        def fit(self, X, y=None):
            self.pre_ = _make_preprocessor(X)
            self.pre_.fit(X, y)
            return self

        def transform(self, X):
            return self.pre_.transform(X)

    model = LGBMRegressor(
        objective="tweedie",
        tweedie_variance_power=tweedie_power,
        random_state=random_state,
    )

    return Pipeline(
        steps=[
            ("cyclical", cyc),
            ("preprocess", _ColumnAwarePreprocess()),
            ("model", model),
        ]
    )
