import pandas as pd
from src.validation.business_rules import non_negative


def test_non_negative():
    s = pd.Series([1, -1, 0])
    out = non_negative(s)
    assert out.iloc[0] == 1
    assert pd.isna(out.iloc[1])
