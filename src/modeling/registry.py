from __future__ import annotations

from pathlib import Path
import json
import joblib


def save_model_artifacts(model, calibrator, feature_columns, model_dir: str = "models") -> None:
    model_path = Path(model_dir)
    model_path.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_path / "xgboost_model.pkl")
    joblib.dump(calibrator, model_path / "calibrator.pkl")
    with open(model_path / "feature_columns.json", "w", encoding="utf-8") as f:
        json.dump(feature_columns, f, indent=2)
