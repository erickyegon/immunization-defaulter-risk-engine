from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.cleaning.standardize import snake_case_columns, standardize_text, parse_datetime, add_missingness_score
from src.validation.schema_checks import require_columns
from src.validation.business_rules import validate_county
from src.utils.io import read_table, write_table

RAW_PATH = Path("data/raw/active_chps.csv")
OUT_PATH = Path("data/interim/slv_chw_registry.csv")


def clean_chw_registry() -> pd.DataFrame:
    df = snake_case_columns(read_table(RAW_PATH))
    require_columns(df, ["chw_uuid", "reported"])
    df["chw_key"] = standardize_text(df["chw_uuid"])
    df["registry_snapshot_datetime"] = parse_datetime(df["reported"])
    df["registry_snapshot_month"] = df["registry_snapshot_datetime"].dt.to_period("M").astype(str)
    for col in ["chw_name", "chw_area_name", "community_unit", "sub_county_name"]:
        if col in df:
            df[col.replace("_name", "") + "_clean" if col.endswith("_name") else f"{col}_clean"] = standardize_text(df[col]).str.lower()
    county_col = "county_name" if "county_name" in df else "county"
    if county_col in df:
        county_result = validate_county(df[county_col])
        df["county_clean"] = county_result["county_clean"]
        df["county_valid_flag"] = county_result["county_valid_flag"]
    df = df.sort_values(["chw_key", "registry_snapshot_datetime"])
    df["chw_registry_record_rank"] = df.groupby("chw_key").cumcount(ascending=True) + 1
    df["is_active_chw"] = 1
    df = add_missingness_score(df)
    write_table(df, OUT_PATH)
    return df
