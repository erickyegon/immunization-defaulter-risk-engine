from __future__ import annotations

import pandas as pd


def subgroup_risk_summary(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    return df.groupby(group_col, dropna=False).agg(
        n=("predicted_risk", "size"),
        mean_predicted_risk=("predicted_risk", "mean"),
    ).reset_index()
