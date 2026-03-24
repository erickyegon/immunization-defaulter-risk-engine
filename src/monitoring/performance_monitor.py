from __future__ import annotations

from pathlib import Path
import pandas as pd

from src.utils.io import read_table

FEATURE_PATH = Path("data/outputs/scored_child_month.csv")
OUT_PATH = Path("reports/deployment_validation/performance_monitor.csv")


def generate_monitoring_report() -> pd.DataFrame:
    df = read_table(FEATURE_PATH)
    report = pd.DataFrame({
        "n_scored": [len(df)],
        "mean_predicted_risk": [df["predicted_risk"].mean() if "predicted_risk" in df else None],
        "pct_high_risk": [(df["predicted_risk"] >= 0.5).mean() if "predicted_risk" in df else None],
    })
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(OUT_PATH, index=False)
    return report
