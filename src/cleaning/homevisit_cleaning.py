from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.cleaning.standardize import snake_case_columns, parse_datetime, add_missingness_score
from src.validation.schema_checks import require_columns
from src.utils.io import read_table, write_table

RAW_PATH = Path("data/raw/homevisit.csv")
OUT_PATH = Path("data/interim/slv_homevisit_events.csv")


def clean_homevisit() -> pd.DataFrame:
    df = snake_case_columns(read_table(RAW_PATH))
    require_columns(df, ["chw_uuid", "reported"])
    df["chw_key"] = df["chw_uuid"].astype("string").str.strip()
    if "family_id" in df:
        df["family_key"] = df["family_id"].astype("string").str.strip()
    df["homevisit_datetime"] = parse_datetime(df["reported"])
    df["year_month"] = df["homevisit_datetime"].dt.to_period("M").astype(str)
    df = add_missingness_score(df)
    write_table(df, OUT_PATH)
    return df
