from __future__ import annotations

import pandas as pd
from src.explainability.reason_codes import derive_reason_codes


def create_risk_list(df: pd.DataFrame, threshold: float = 0.5) -> pd.DataFrame:
    out = df.copy()
    out = out[out["predicted_risk"] >= threshold].copy()
    out["risk_tier"] = pd.cut(
        out["predicted_risk"],
        bins=[0, 0.5, 0.75, 1.0],
        labels=["medium", "high", "critical"],
        include_lowest=True,
    )
    out["reason_codes"] = out.apply(derive_reason_codes, axis=1)
    return out.sort_values("predicted_risk", ascending=False)
