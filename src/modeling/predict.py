from __future__ import annotations

import json
from pathlib import Path
import joblib
import pandas as pd


def load_artifacts(model_dir: str = "models"):
    model_path = Path(model_dir)
    model = joblib.load(model_path / "xgboost_model.pkl")
    calibrator = joblib.load(model_path / "calibrator.pkl")
    feature_columns = json.loads((model_path / "feature_columns.json").read_text(encoding="utf-8"))
    return model, calibrator, feature_columns


def predict_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    _, calibrator, feature_columns = load_artifacts()
    X = df.reindex(columns=feature_columns)
    out = df.copy()
    out["predicted_risk"] = calibrator.predict_proba(X)[:, 1]
    return out
