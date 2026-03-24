from __future__ import annotations

from pathlib import Path
import matplotlib.pyplot as plt
import shap
from src.utils.io import read_table
from src.modeling.predict import load_artifacts

FEATURE_PATH = Path("data/processed/gld_model_features_child_month.csv")
OUT_PATH = Path("reports/shap/shap_summary.png")


def generate_global_shap() -> None:
    model, _, feature_columns = load_artifacts()
    df = read_table(FEATURE_PATH)
    X = df.reindex(columns=feature_columns)
    preprocessor = model.named_steps["preprocessor"]
    booster = model.named_steps["model"]
    X_t = preprocessor.transform(X)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_t)
    plt.figure()
    shap.summary_plot(shap_values, X_t, show=False)
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(OUT_PATH, bbox_inches="tight")
    plt.close()
