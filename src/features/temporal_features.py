from __future__ import annotations

import pandas as pd


def add_temporal_features(df: pd.DataFrame, time_col: str = "index_date") -> pd.DataFrame:
    out = df.copy()
    dt = pd.to_datetime(out[time_col], errors="coerce", utc=True)
    out["index_month_num"] = dt.dt.month
    out["index_quarter"] = dt.dt.quarter
    out["index_year"] = dt.dt.year
    return out
