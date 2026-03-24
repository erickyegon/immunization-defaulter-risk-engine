from __future__ import annotations

import pandas as pd


def compare_feature_means(reference_df: pd.DataFrame, current_df: pd.DataFrame) -> pd.DataFrame:
    numeric_cols = [c for c in current_df.columns if pd.api.types.is_numeric_dtype(current_df[c]) and c in reference_df.columns]
    return pd.DataFrame({
        "feature": numeric_cols,
        "reference_mean": [reference_df[c].mean() for c in numeric_cols],
        "current_mean": [current_df[c].mean() for c in numeric_cols],
    })
