from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.utils.io import read_table

POP_PATH = Path("data/interim/slv_population_context.csv")
CHW_PATH = Path("data/interim/slv_chw_registry.csv")


def build_area_context() -> pd.DataFrame:
    pop = read_table(POP_PATH)
    chw = read_table(CHW_PATH)
    latest_chw = chw.sort_values(["chw_key", "registry_snapshot_datetime"]).groupby("chw_key", as_index=False).tail(1)
    join_cols = [c for c in ["chw_key", "county_clean", "sub_county_clean"] if c in latest_chw.columns]
    merged = pop.merge(latest_chw[join_cols], on="chw_key", how="left") if "chw_key" in pop and "chw_key" in latest_chw else pop.copy()
    out = merged.groupby([c for c in ["county_clean", "sub_county_clean"] if c in merged.columns], as_index=False).agg(
        u2_population_context=("u2_pop_clean", "max"),
        u5_population_context=("u5_pop_clean", "max"),
        wra_population_context=("wra_pop_clean", "max"),
    )
    return out
