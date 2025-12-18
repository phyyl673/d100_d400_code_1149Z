from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from sklearn.exceptions import NotFittedError

from bike_demand.feature_engineering import CyclicalEncoder


@pytest.mark.parametrize(
    "col,period,values,drop_original",
    [
        ("hour", 24, [0, 6, 12, 18, 23], False),
        ("month", 12, [1, 3, 6, 9, 12], False),
        ("hour", 24, [0, 1, 2], True),
    ],
)
def test_cyclical_encoder_adds_columns_and_unit_circle(col, period, values, drop_original):
    df = pd.DataFrame({col: values, "other": [10] * len(values)})

    enc = CyclicalEncoder(
        columns=[col],
        periods={col: period},
        drop_original=drop_original,
    )
    out = enc.fit_transform(df)

    sin_col = f"{col}_sin"
    cos_col = f"{col}_cos"

    assert sin_col in out.columns
    assert cos_col in out.columns
    assert len(out) == len(df)

    # original column kept/dropped
    if drop_original:
        assert col not in out.columns
    else:
        assert col in out.columns

    # check sin^2 + cos^2 ~= 1
    r2 = out[sin_col].to_numpy() ** 2 + out[cos_col].to_numpy() ** 2
    assert np.allclose(r2, 1.0, atol=1e-10)


def test_cyclical_encoder_get_feature_names_out():
    df = pd.DataFrame({"hour": [0, 1, 2], "x": [1.0, 2.0, 3.0]})
    enc = CyclicalEncoder(columns=["hour"], periods={"hour": 24}, drop_original=False)
    enc.fit(df)

    names = enc.get_feature_names_out()
    assert "hour" in names
    assert "hour_sin" in names
    assert "hour_cos" in names
    assert "x" in names


def test_transform_raises_if_not_fitted():
    df = pd.DataFrame({"hour": [0, 1, 2]})
    enc = CyclicalEncoder(columns=["hour"], periods={"hour": 24})
    with pytest.raises(NotFittedError):
        enc.transform(df)


def test_fit_raises_for_missing_column():
    df = pd.DataFrame({"hour": [0, 1, 2]})
    enc = CyclicalEncoder(columns=["month"], periods={"month": 12})
    with pytest.raises(KeyError):
        enc.fit(df)


def test_fit_raises_for_missing_period():
    df = pd.DataFrame({"hour": [0, 1, 2]})
    enc = CyclicalEncoder(columns=["hour"], periods={})
    with pytest.raises(KeyError):
        enc.fit(df)


@pytest.mark.parametrize("bad_period", [0, -1, -24])
def test_fit_raises_for_nonpositive_period(bad_period):
    df = pd.DataFrame({"hour": [0, 1, 2]})
    enc = CyclicalEncoder(columns=["hour"], periods={"hour": bad_period})
    with pytest.raises(ValueError):
        enc.fit(df)


def test_fit_raises_for_nonnumeric_period():
    df = pd.DataFrame({"hour": [0, 1, 2]})
    enc = CyclicalEncoder(columns=["hour"], periods={"hour": "24"})
    with pytest.raises(TypeError):
        enc.fit(df)
