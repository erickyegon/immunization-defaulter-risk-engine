from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.cleaning.standardize import snake_case_columns, standardize_boolean, parse_datetime, add_missingness_score
from src.validation.schema_checks import require_columns
from src.validation.business_rules import validate_county, non_negative
from src.utils.constants import CORE_VACCINE_COLUMNS
from src.utils.io import read_table, write_table

RAW_PATH = Path("data/raw/iz.csv")
OUT_PATH = Path("data/interim/slv_iz_events.csv")


REQUIRED_COLUMNS = ["uuid", "patient_id", "reported"]


def clean_iz() -> pd.DataFrame:
    df = read_table(RAW_PATH)
    df = snake_case_columns(df)
    require_columns(df, REQUIRED_COLUMNS)

    if "reported" in df:
        df["event_datetime"] = parse_datetime(df["reported"])
        df["event_date"] = df["event_datetime"].dt.date
        df["year_month"] = df["event_datetime"].dt.to_period("M").astype(str)

    df["iz_event_key"] = df["uuid"].astype("string").str.strip()
    df["child_key"] = df["patient_id"].astype("string").str.strip()

    if "patient_sex" in df:
        df["sex"] = df["patient_sex"].astype("string").str.strip().str.lower()
        df["sex"] = df["sex"].replace({"m": "male", "f": "female"})

    for age_col in ["patient_age_in_days", "patient_age_in_months", "patient_age_in_years"]:
        if age_col in df:
            df[age_col] = non_negative(df[age_col])

    if "patient_age_in_days" in df:
        df["age_days"] = df["patient_age_in_days"]
        df["age_months_rebuilt"] = (df["age_days"] / 30.4375).round(1)
        df["age_band"] = pd.cut(
            df["age_days"],
            bins=[-1, 41, 69, 97, 269, 364, 729, 1825],
            labels=["0-6w", "6-10w", "10-14w", "14w-9m", "9-12m", "12-24m", "24m+"],
        )
        if "patient_age_in_months" in df:
            raw_months = pd.to_numeric(df["patient_age_in_months"], errors="coerce")
            df["age_consistency_flag"] = ((raw_months - df["age_months_rebuilt"]).abs() <= 1).fillna(False).astype(int)
        else:
            df["age_consistency_flag"] = pd.NA

    if "county" in df:
        county_result = validate_county(df["county"])
        df["county_raw"] = df["county"]
        df["county_clean"] = county_result["county_clean"]
        df["county_valid_flag"] = county_result["county_valid_flag"]

    if "accuracy" in df:
        df["gps_accuracy_m"] = pd.to_numeric(df["accuracy"], errors="coerce")
        df["gps_reliable_flag"] = (df["gps_accuracy_m"] <= 100).fillna(False).astype(int)

    boolean_candidates = [
        c for c in df.columns if c.startswith("has_") or c.startswith("needs_") or c.startswith("is_")
    ] + ["visited", "client_available"]
    for col in set(boolean_candidates):
        if col in df:
            df[col] = standardize_boolean(df[col])

    for col in CORE_VACCINE_COLUMNS:
        if col in df:
            df[col] = standardize_boolean(df[col]).fillna(0).astype(int)
        else:
            df[col] = 0

    if "due_count" in df:
        df["due_count_raw"] = pd.to_numeric(df["due_count"], errors="coerce")
        df["due_count_raw"] = df["due_count_raw"].where(df["due_count_raw"] >= 0)

    if "follow_up_date" in df:
        df["follow_up_date"] = parse_datetime(df["follow_up_date"])
    if "immunization_follow_up_date" in df:
        df["immunization_follow_up_date"] = parse_datetime(df["immunization_follow_up_date"])

    df = add_missingness_score(df)
    write_table(df, OUT_PATH)
    return df
