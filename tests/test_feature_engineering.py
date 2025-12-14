import pandas as pd
import pytest

from bike_demand.feature_engineering import DateFeaturesExtractor


@pytest.mark.parametrize(
    "dates, expected_months",
    [
        (["01/12/2017", "02/12/2017"], [12, 12]),
        (["15/01/2018", "20/02/2018"], [1, 2]),
    ],
)
def test_date_features_extractor_basic(dates, expected_months):
    df = pd.DataFrame({"date": pd.to_datetime(dates, dayfirst=True)})
    tr = DateFeaturesExtractor(date_col="date")
    out = tr.fit_transform(df)

    assert list(out.columns) == ["month", "dayofweek", "is_weekend"]
    assert out["month"].tolist() == expected_months
    assert out.shape[0] == len(dates)


def test_date_features_extractor_missing_column_raises():
    df = pd.DataFrame({"not_date": ["2017-12-01"]})
    tr = DateFeaturesExtractor(date_col="date")
    with pytest.raises(KeyError):
        tr.fit_transform(df)
