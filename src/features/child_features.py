from __future__ import annotations

import pandas as pd


def build_child_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["prior_follow_up_flag_count"] = 0
    out["prior_default_flag_count"] = 0
    out["days_since_last_iz_event"] = 0
    out["core_fields_complete_flag"] = out[["child_key", "index_month", "age_days"]].notna().all(axis=1).astype(int)
    return out
