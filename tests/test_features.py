import pandas as pd
from src.features.temporal_features import add_temporal_features


def test_add_temporal_features():
    df = pd.DataFrame({"index_date": ["2025-01-31"]})
    out = add_temporal_features(df)
    assert "index_month_num" in out.columns
