from __future__ import annotations

import pandas as pd
from src.utils.constants import VALID_KENYA_COUNTIES


def validate_county(series: pd.Series) -> pd.DataFrame:
    cleaned = series.astype(str).str.strip().str.lower()
    return pd.DataFrame({
        "county_clean": cleaned.where(cleaned.isin(VALID_KENYA_COUNTIES)),
        "county_valid_flag": cleaned.isin(VALID_KENYA_COUNTIES).astype(int),
    })


def non_negative(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.where(numeric >= 0)
