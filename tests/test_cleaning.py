import pandas as pd
from src.cleaning.standardize import standardize_boolean


def test_standardize_boolean():
    s = pd.Series(["yes", "no", "true", "0", None])
    out = standardize_boolean(s)
    assert out.iloc[0] == 1
    assert out.iloc[1] == 0
    assert out.iloc[2] == 1
    assert out.iloc[3] == 0
