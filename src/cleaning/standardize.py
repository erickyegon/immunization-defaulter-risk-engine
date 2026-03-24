from __future__ import annotations

import re
import pandas as pd
from src.utils.constants import BOOLEAN_TRUE_VALUES, BOOLEAN_FALSE_VALUES
from src.utils.dates import to_datetime


def snake_case_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [re.sub(r"[^a-z0-9]+", "_", c.strip().lower()).strip("_") for c in out.columns]
    return out


def standardize_boolean(series: pd.Series) -> pd.Series:
    def _map(value):
        if pd.isna(value):
            return pd.NA
        norm = str(value).strip().lower()
        if norm in BOOLEAN_TRUE_VALUES:
            return 1
        if norm in BOOLEAN_FALSE_VALUES:
            return 0
        return pd.NA
    return series.map(_map).astype("Int64")


def standardize_text(series: pd.Series) -> pd.Series:
    return series.astype("string").str.strip()


def parse_datetime(series: pd.Series) -> pd.Series:
    return to_datetime(series)


def add_missingness_score(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["row_missingness_pct"] = out.isna().mean(axis=1)
    return out
