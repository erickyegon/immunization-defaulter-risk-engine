from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.cleaning.standardize import snake_case_columns, add_missingness_score
from src.validation.business_rules import non_negative
from src.utils.io import read_table, write_table

RAW_PATH = Path("data/raw/population.csv")
OUT_PATH = Path("data/interim/slv_population_context.csv")


def clean_population() -> pd.DataFrame:
    df = snake_case_columns(read_table(RAW_PATH))
    if "chw_uuid" in df:
        df["chw_key"] = df["chw_uuid"].astype("string").str.strip()
    for col in ["u2_pop", "u5_pop", "wra_pop"]:
        if col in df:
            df[f"{col}_clean"] = non_negative(df[col])
    valid_cols = [c for c in ["u2_pop_clean", "u5_pop_clean", "wra_pop_clean"] if c in df]
    if valid_cols:
        df["population_context_valid_flag"] = df[valid_cols].notna().any(axis=1).astype(int)
    df = add_missingness_score(df)
    write_table(df, OUT_PATH)
    return df
