from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.utils.io import read_table

SUP_PATH = Path("data/interim/slv_supervision_events.csv")
HV_PATH = Path("data/interim/slv_homevisit_events.csv")
CHW_PATH = Path("data/interim/slv_chw_registry.csv")


def build_chw_context() -> pd.DataFrame:
    sup = read_table(SUP_PATH)
    hv = read_table(HV_PATH)
    chw = read_table(CHW_PATH)

    if "year_month" not in sup:
        sup["year_month"] = pd.to_datetime(sup["supervision_datetime"], errors="coerce", utc=True).dt.to_period("M").astype(str)
    if "year_month" not in hv:
        hv["year_month"] = pd.to_datetime(hv["homevisit_datetime"], errors="coerce", utc=True).dt.to_period("M").astype(str)

    sup_agg = sup.groupby(["chw_key", "year_month"], as_index=False).agg(
        supervision_events_last_90d=("supervision_event_key", "count"),
        mean_immunization_score_90d=("calc_immunization_score", "mean"),
        pct_tools_available_90d=("has_all_tools", "mean"),
        days_since_last_supervision=("supervision_datetime", lambda s: 0 if s.notna().any() else None),
    )
    hv_agg = hv.groupby(["chw_key", "year_month"], as_index=False).agg(
        homevisit_count_30d=("chw_key", "count"),
        homevisit_count_90d=("chw_key", "count"),
    )
    latest_chw = chw.sort_values(["chw_key", "registry_snapshot_datetime"]).groupby("chw_key", as_index=False).tail(1)
    latest_chw = latest_chw[[c for c in ["chw_key", "county_clean", "sub_county_clean", "chw_area_clean", "community_unit_clean"] if c in latest_chw.columns]]
    out = sup_agg.merge(hv_agg, on=["chw_key", "year_month"], how="outer").merge(latest_chw, on="chw_key", how="left")
    out = out.rename(columns={"year_month": "index_month"})
    return out
