from __future__ import annotations

from pathlib import Path
import json
from src.utils.io import read_table, write_table
from src.modeling.predict import predict_dataframe
from src.serving.risk_lists import create_risk_list
from src.serving.export_outputs import export_dataframe

FEATURE_PATH = Path("data/processed/gld_model_features_child_month.csv")
SCORED_PATH = Path("data/outputs/scored_child_month.csv")
RISK_PATH = Path("data/outputs/high_risk_children.csv")
TRAIN_METRICS = Path("reports/model_performance/train_metrics.json")


def batch_score_current_cohort():
    df = read_table(FEATURE_PATH)
    scored = predict_dataframe(df)
    write_table(scored, SCORED_PATH)
    threshold = 0.5
    if TRAIN_METRICS.exists():
        with open(TRAIN_METRICS, "r", encoding="utf-8") as f:
            threshold = json.load(f).get("threshold", 0.5)
    risk_list = create_risk_list(scored, threshold=threshold)
    export_dataframe(risk_list, RISK_PATH)
    return risk_list
