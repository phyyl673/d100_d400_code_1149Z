from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_is_fitted


@dataclass
class CyclicalEncoder(BaseEstimator, TransformerMixin):
    """
    Encode cyclical (periodic) features into sine/cosine components.
    
    This transformer is particularly useful for time-based features like:
    - Hour (0-23): 23 and 0 are only 1 hour apart
    - Month (1-12): December and January are adjacent
    - Day of week (0-6): Sunday and Monday are adjacent

    Parameters
    ----------
    columns : Iterable[str]
        Column names to encode (e.g. ["hour", "month"]).
    periods : dict[str, int]
        Period per column (e.g. {"hour": 24, "month": 12}).
    drop_original : bool, optional
        If True, drop original columns after encoding. By default False.

    Notes
    -----
    - Expects pandas DataFrame input to preserve column names.
    - Validates periods during fit().
    """

    columns: Iterable[str]
    periods: dict[str, int]
    drop_original: bool = False

    # fitted attributes (IMPORTANT: created only after fit)
    feature_names_in_: np.ndarray = field(init=False, repr=False)

    def fit(self, X, y=None):
        X_df = _ensure_dataframe(X)
        cols = list(self.columns)

        # Validate requested columns exist
        missing_cols = [c for c in cols if c not in X_df.columns]
        if missing_cols:
            raise KeyError(f"CyclicalEncoder: missing column(s): {missing_cols}")

        # Validate periods
        for c in cols:
            if c not in self.periods:
                raise KeyError(f"CyclicalEncoder: missing period for '{c}'")
            p = self.periods[c]
            if not isinstance(p, (int, float)):
                raise TypeError(
                    f"CyclicalEncoder: period for '{c}' must be numeric, got {type(p).__name__}"
                )
            if p <= 0:
                raise ValueError(
                    f"CyclicalEncoder: period for '{c}' must be > 0, got {p}"
                )

        # record input features for sklearn compatibility
        self.feature_names_in_ = np.array(list(X_df.columns), dtype=object)

        return self

    def transform(self, X):
        check_is_fitted(self, attributes=["feature_names_in_"])

        X_df = _ensure_dataframe(X)
        cols = list(self.columns)

        missing_cols = [c for c in cols if c not in X_df.columns]
        if missing_cols:
            raise KeyError(f"CyclicalEncoder: missing column(s): {missing_cols}")

        out = X_df.copy()

        for c in cols:
            period = float(self.periods[c])
            vals = pd.to_numeric(out[c], errors="raise").astype(float).to_numpy()
            angle = 2.0 * np.pi * vals / period

            out[f"{c}_sin"] = np.sin(angle)
            out[f"{c}_cos"] = np.cos(angle)

            if self.drop_original:
                out = out.drop(columns=[c])

        return out

    def get_feature_names_out(self, input_features=None):
        """
        Return output feature names for integration with sklearn.
        """
        check_is_fitted(self, attributes=["feature_names_in_"])

        if input_features is None:
            in_feats = list(self.feature_names_in_)
        else:
            in_feats = list(input_features)

        out_feats = list(in_feats)
        cols = list(self.columns)

        for c in cols:
            out_feats.append(f"{c}_sin")
            out_feats.append(f"{c}_cos")
            if self.drop_original and c in out_feats:
                out_feats.remove(c)

        return np.array(out_feats, dtype=object)


def _ensure_dataframe(X) -> pd.DataFrame:
    if isinstance(X, pd.DataFrame):
        return X
    raise TypeError(
        "CyclicalEncoder expects a pandas DataFrame input "
        f"(got {type(X).__name__})."
    )
