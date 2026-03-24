from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.cleaning.standardize import snake_case_columns, standardize_boolean, parse_datetime, add_missingness_score
from src.utils.io import read_table, write_table
from src.validation.schema_checks import require_columns

RAW_PATH = Path("data/raw/supervision.csv")
OUT_PATH = Path("data/interim/slv_supervision_events.csv")


def clean_supervision() -> pd.DataFrame:
    df = snake_case_columns(read_table(RAW_PATH))
    require_columns(df, ["uuid", "chw_uuid", "reported"])
    df["supervision_event_key"] = df["uuid"].astype("string").str.strip()
    df["chw_key"] = df["chw_uuid"].astype("string").str.strip()
    df["supervision_datetime"] = parse_datetime(df["reported"])
    df["year_month"] = df["supervision_datetime"].dt.to_period("M").astype(str)
    for col in ["last_visit_date", "next_supervision_visit_date"]:
        if col in df:
            df[col] = parse_datetime(df[col])
    for col in [
        "has_all_tools", "has_proper_protective_equipment", "has_essential_medicines",
        "calc_has_all_tools", "calc_has_proper_protective_equipment",
    ]:
        if col in df:
            df[col] = standardize_boolean(df[col])
    for col in [
        "calc_assessment_score", "calc_assessment_denominator",
        "calc_immunization_score", "calc_immunization_denominator",
    ]:
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    if {"calc_assessment_score", "calc_assessment_denominator"}.issubset(df.columns):
        df["assessment_score_valid_flag"] = (df["calc_assessment_denominator"] > 0).fillna(False).astype(int)
    if {"calc_immunization_score", "calc_immunization_denominator"}.issubset(df.columns):
        df["immunization_score_valid_flag"] = (df["calc_immunization_denominator"] > 0).fillna(False).astype(int)
    df = add_missingness_score(df)
    write_table(df, OUT_PATH)
    return df
