from __future__ import annotations

import pandas as pd


def simple_missingness_profile(df: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame({
        "column": df.columns,
        "missing_pct": [df[c].isna().mean() for c in df.columns],
        "n_unique": [df[c].nunique(dropna=True) for c in df.columns],
    })
