from __future__ import annotations

from pathlib import Path
from src.utils.io import read_table, write_table
from src.features.child_features import build_child_features
from src.features.chw_context_features import build_chw_context
from src.features.area_context_features import build_area_context
from src.features.temporal_features import add_temporal_features

COHORT_LABEL_PATH = Path("data/processed/gld_child_month_labels.csv")
OUT_PATH = Path("data/processed/gld_model_features_child_month.csv")


def build_feature_store():
    base = read_table(COHORT_LABEL_PATH)
    base = build_child_features(base)
    base = add_temporal_features(base, time_col="index_date")

    chw_context = build_chw_context()
    area_context = build_area_context()

    if "chw_key" not in base.columns:
        base["chw_key"] = None
    if "index_month" not in base.columns:
        raise ValueError("Feature base must contain index_month")

    feature_df = base.merge(chw_context, on=["chw_key", "index_month"], how="left")
    area_merge_cols = [c for c in ["county_clean", "sub_county_clean"] if c in feature_df.columns and c in area_context.columns]
    if area_merge_cols:
        feature_df = feature_df.merge(area_context, on=area_merge_cols, how="left")

    write_table(feature_df, OUT_PATH)
    return feature_df
