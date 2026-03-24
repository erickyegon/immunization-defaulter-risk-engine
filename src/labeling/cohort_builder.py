from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.io import read_table, write_table

IN_PATH = Path("data/interim/slv_iz_events.csv")
OUT_PATH = Path("data/processed/gld_child_month_cohort.csv")


def build_child_month_cohort() -> pd.DataFrame:
    df = read_table(IN_PATH)
    cohort = (
        df.sort_values(["child_key", "event_datetime"])
        .groupby(["child_key", "year_month"], as_index=False)
        .agg(
            index_date=("event_datetime", "max"),
            sex=("sex", "last"),
            age_days=("age_days", "max"),
            county_clean=("county_clean", "last"),
            county_valid_flag=("county_valid_flag", "max"),
            row_missingness_pct=("row_missingness_pct", "mean"),
        )
        .rename(columns={"year_month": "index_month"})
    )
    write_table(cohort, OUT_PATH)
    return cohort
