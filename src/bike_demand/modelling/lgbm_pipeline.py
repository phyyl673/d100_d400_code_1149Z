from __future__ import annotations

from typing import Optional

import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from bike_demand.feature_engineering.transformers import CyclicalEncoder
from bike_demand.modelling.glm_pipeline import BikeFeatureSpec


def split_xy(df: pd.DataFrame, spec: BikeFeatureSpec | None = None) -> tuple[pd.DataFrame, pd.Series]:
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

        cyc_num = []
        for c in spec.cyclical:
            s, co = f"{c}_sin", f"{c}_cos"
            if s in cols and co in cols:
                cyc_num += [s, co]

        numeric_features = numeric_base + cyc_num

        num_pipe = Pipeline([("impute", SimpleImputer(strategy="median"))])
        cat_pipe = Pipeline(
            [
                ("impute", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore", drop="first")),
            ]
        )

        return ColumnTransformer(
            transformers=[
                ("num", num_pipe, numeric_features),
                ("cat", cat_pipe, categorical_base),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )

    class _ColumnAwarePreprocess:
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
