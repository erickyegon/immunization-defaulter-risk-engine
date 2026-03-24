from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.labeling.vaccine_schedule import expected_vaccines_by_age
from src.utils.constants import CORE_VACCINE_COLUMNS
from src.utils.io import read_table, write_table

IZ_PATH = Path("data/interim/slv_iz_events.csv")
COHORT_PATH = Path("data/processed/gld_child_month_cohort.csv")
OUT_PATH = Path("data/processed/gld_child_month_labels.csv")


def _compute_due_features(row: pd.Series) -> tuple[list[str], int, int]:
    expected = expected_vaccines_by_age(row.get("age_days"))
    received = [v for v in expected if row.get(v, 0) == 1]
    due = [v for v in expected if v not in received]
    return due, len(received), len(due)


def build_labels() -> pd.DataFrame:
    iz = read_table(IZ_PATH)
    cohort = read_table(COHORT_PATH)

    latest = (
        iz.sort_values(["child_key", "event_datetime"])
        .groupby(["child_key", "year_month"], as_index=False)
        .tail(1)
        .rename(columns={"year_month": "index_month"})
    )
    for col in CORE_VACCINE_COLUMNS:
        if col not in latest:
            latest[col] = 0

    label_df = cohort.merge(latest[["child_key", "index_month", *CORE_VACCINE_COLUMNS]], on=["child_key", "index_month"], how="left")
    due_info = label_df.apply(_compute_due_features, axis=1, result_type="expand")
    label_df["due_vaccines_rebuilt"] = due_info[0].apply(lambda x: ";".join(x))
    label_df["num_received_vaccines"] = due_info[1]
    label_df["num_due_vaccines_rebuilt"] = due_info[2]
    label_df["num_overdue_vaccines"] = label_df["num_due_vaccines_rebuilt"]

    # Conservative portfolio label: if child has one or more due vaccines at index month, flag as default candidate.
    # Replace with true forward-looking outcome once complete longitudinal event history is available.
    label_df["default_30d"] = (label_df["num_due_vaccines_rebuilt"] > 0).astype(int)
    label_df["fully_immunized_for_age_gap"] = (label_df["num_due_vaccines_rebuilt"] == 0).astype(int)
    label_df["dropout_risk"] = ((label_df.get("has_penta_1", 0) == 1) & (label_df.get("has_penta_3", 0) == 0)).astype(int)

    write_table(label_df, OUT_PATH)
    return label_df
