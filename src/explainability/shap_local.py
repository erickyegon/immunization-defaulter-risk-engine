from __future__ import annotations

import pandas as pd
import shap
from src.modeling.predict import load_artifacts


def explain_single_child(feature_row: pd.DataFrame) -> pd.DataFrame:
    model, _, feature_columns = load_artifacts()
    X = feature_row.reindex(columns=feature_columns)
    preprocessor = model.named_steps["preprocessor"]
    booster = model.named_steps["model"]
    X_t = preprocessor.transform(X)
    explainer = shap.TreeExplainer(booster)
    shap_values = explainer.shap_values(X_t)
    return pd.DataFrame({
        "feature": feature_columns[: len(shap_values[0])],
        "shap_value": list(shap_values[0])[: len(feature_columns)],
    }).sort_values("shap_value", ascending=False)
