from __future__ import annotations

from pathlib import Path
import json
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

from src.utils.io import read_table
from src.modeling.evaluate import evaluate_predictions
from src.modeling.calibration import calibrate_model
from src.modeling.thresholding import top_k_threshold
from src.modeling.registry import save_model_artifacts

FEATURE_PATH = Path("data/processed/gld_model_features_child_month.csv")
REPORT_PATH = Path("reports/model_performance/train_metrics.json")


def temporal_split(df: pd.DataFrame, time_col: str = "index_month"):
    months = sorted(df[time_col].dropna().unique().tolist())
    if len(months) < 3:
        raise ValueError("Need at least 3 distinct months for temporal split")
    train_months = months[:-2]
    valid_months = [months[-2]]
    test_months = [months[-1]]
    train_df = df[df[time_col].isin(train_months)].copy()
    valid_df = df[df[time_col].isin(valid_months)].copy()
    test_df = df[df[time_col].isin(test_months)].copy()
    return train_df, valid_df, test_df


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]
    return ColumnTransformer(
        transformers=[
            ("num", Pipeline([("imputer", SimpleImputer(strategy="median"))]), numeric_cols),
            ("cat", Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("onehot", OneHotEncoder(handle_unknown="ignore")),
            ]), categorical_cols),
        ]
    )


def train_pipeline() -> dict:
    df = read_table(FEATURE_PATH)
    target_col = "default_30d"
    drop_cols = [target_col, "child_key", "index_month", "index_date", "due_vaccines_rebuilt"]
    drop_cols = [c for c in drop_cols if c in df.columns]

    train_df, valid_df, test_df = temporal_split(df)
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df[target_col].astype(int)
    X_valid = valid_df.drop(columns=drop_cols)
    y_valid = valid_df[target_col].astype(int)
    X_test = test_df.drop(columns=drop_cols)
    y_test = test_df[target_col].astype(int)

    preprocessor = build_preprocessor(X_train)

    baseline = Pipeline([
        ("preprocessor", preprocessor),
        ("model", LogisticRegression(max_iter=1000, class_weight="balanced")),
    ])
    baseline.fit(X_train, y_train)

    xgb = Pipeline([
        ("preprocessor", preprocessor),
        ("model", XGBClassifier(
            objective="binary:logistic",
            eval_metric="aucpr",
            max_depth=4,
            learning_rate=0.05,
            n_estimators=300,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2.0,
            random_state=42,
        )),
    ])
    xgb.fit(X_train, y_train)

    valid_probs = xgb.predict_proba(X_valid)[:, 1]
    test_probs = xgb.predict_proba(X_test)[:, 1]
    calibrator = calibrate_model(xgb, X_valid, y_valid)
    calibrated_test_probs = calibrator.predict_proba(X_test)[:, 1]

    metrics = {
        "baseline_valid": evaluate_predictions(y_valid, baseline.predict_proba(X_valid)[:, 1]),
        "xgb_valid": evaluate_predictions(y_valid, valid_probs),
        "xgb_test_calibrated": evaluate_predictions(y_test, calibrated_test_probs),
        "threshold": top_k_threshold(calibrated_test_probs, fraction=0.2),
        "n_train": len(train_df),
        "n_valid": len(valid_df),
        "n_test": len(test_df),
    }

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)

    save_model_artifacts(xgb, calibrator, X_train.columns.tolist())
    return metrics
