from __future__ import annotations

import pandas as pd


def derive_reason_codes(row: pd.Series) -> list[str]:
    reasons = []
    if row.get("num_due_vaccines_rebuilt", 0) > 0:
        reasons.append("child currently has due vaccines outstanding")
    if row.get("days_since_last_supervision") not in [None, pd.NA] and pd.notna(row.get("days_since_last_supervision")):
        if row.get("days_since_last_supervision", 0) > 60:
            reasons.append("long gap since recent CHW supervision")
    if row.get("homevisit_count_30d", 0) == 0:
        reasons.append("no recent home visit activity captured")
    if row.get("county_valid_flag", 1) == 0:
        reasons.append("source data contains administrative field inconsistency")
    return reasons[:3]
