"""
model/trainer.py
─────────────────────────────────────────────────────────────────────────────
XGBoost training pipeline with:
  - Stratified train/test split
  - Class imbalance handling via scale_pos_weight
  - Probability calibration (isotonic)
  - Full MLflow experiment tracking
  - Model artifact persistence
"""

import logging
import os
import joblib
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml
from sklearn.calibration import CalibratedClassifierCV
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from xgboost import XGBClassifier

logger = logging.getLogger(__name__)


class IZDefaulterTrainer:
    """
    End-to-end XGBoost training with MLflow tracking.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.model_cfg       = self.cfg["model"]
        self.eval_cfg        = self.cfg["evaluation"]
        self.mlflow_cfg      = self.cfg["mlflow"]
        self.calibration_cfg = self.cfg["calibration"]

        self.preprocessor    = None
        self.model           = None
        self.calibrated_model = None
        self.feature_names:  List[str] = []
        self.run_id:         Optional[str] = None

    # ── Main training entry ────────────────────────────────────────────────

    def train(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        preprocessor,
        feature_names: List[str],
        run_name: Optional[str] = None,
    ) -> Dict:
        """
        Train XGBoost, calibrate probabilities, log to MLflow.

        Returns
        -------
        dict with keys: model, calibrated_model, metrics, feature_names
        """
        self.preprocessor  = preprocessor
        self.feature_names = feature_names

        # Setup MLflow
        mlflow.set_tracking_uri(self.mlflow_cfg["tracking_uri"])
        mlflow.set_experiment(self.mlflow_cfg["experiment_name"])

        with mlflow.start_run(run_name=run_name or "xgb_iz_defaulter") as run:
            self.run_id = run.info.run_id
            logger.info(f"\nMLflow run: {self.run_id}")

            # Split
            X_train, X_test, y_train, y_test = train_test_split(
                X, y,
                test_size   = self.eval_cfg["test_size"],
                stratify    = y if self.eval_cfg["stratified"] else None,
                random_state= 42,
            )
            logger.info(f"  Train: {len(X_train)} | Test: {len(X_test)}")

            # Preprocess
            X_train_t = preprocessor.fit_transform(X_train)
            X_test_t  = preprocessor.transform(X_test)

            # Derive actual feature names from the FITTED preprocessor.
            # fp._all_feature_cols() returns ALL defined features (including
            # all-null ones that build_preprocessor silently dropped), so the
            # passed-in feature_names list may be longer than the matrix width.
            # Using transformers_ ensures perfect alignment with column order.
            actual_feature_names = []
            for _name, _trans, _cols in preprocessor.transformers_:
                if _name != "remainder":
                    actual_feature_names.extend(_cols)
            if actual_feature_names:
                self.feature_names = actual_feature_names
                if len(actual_feature_names) != len(feature_names):
                    logger.info(
                        f"  Feature names adjusted: {len(feature_names)} defined → "
                        f"{len(actual_feature_names)} in preprocessor "
                        f"({len(feature_names) - len(actual_feature_names)} all-null dropped)"
                    )

            # Class imbalance weight
            n_neg = int((y_train == 0).sum())
            n_pos = int((y_train == 1).sum())
            spw   = n_neg / max(n_pos, 1)
            logger.info(f"  Class balance: {n_neg} neg / {n_pos} pos → scale_pos_weight={spw:.2f}")

            # XGBoost params
            params = {**self.model_cfg["base_params"], "scale_pos_weight": spw}
            # Remove non-XGBoost keys
            train_params = {k: v for k, v in params.items()
                           if k not in ("use_label_encoder",)}

            # Build XGBoost model
            xgb = XGBClassifier(
                **{k: v for k, v in train_params.items()
                   if k not in ("early_stopping_rounds",)},
                early_stopping_rounds=train_params.get("early_stopping_rounds", 50),
                verbosity=0,
            )

            # Fit with early stopping
            xgb.fit(
                X_train_t, y_train,
                eval_set=[(X_test_t, y_test)],
                verbose=False,
            )
            self.model = xgb
            best_iter  = xgb.best_iteration
            logger.info(f"  Best iteration: {best_iter}")

            # Calibrate probabilities
            if self.calibration_cfg["enabled"] and len(X_train) >= 20:
                logger.info("  Calibrating probabilities (isotonic)...")
                # Clone xgb WITHOUT early_stopping (CalibratedCV provides no eval_set)
                xgb_for_cal = XGBClassifier(
                    **{k: v for k, v in train_params.items()
                       if k not in ("early_stopping_rounds", "scale_pos_weight")},
                    scale_pos_weight=spw,
                    verbosity=0,
                )
                self.calibrated_model = CalibratedClassifierCV(
                    xgb_for_cal,
                    method = self.calibration_cfg["method"],
                    cv     = min(self.calibration_cfg["cv"], n_pos, n_neg),
                )
                self.calibrated_model.fit(X_train_t, y_train)
                y_prob = self.calibrated_model.predict_proba(X_test_t)[:, 1]
            else:
                self.calibrated_model = xgb
                y_prob = xgb.predict_proba(X_test_t)[:, 1]

            y_pred = (y_prob >= 0.5).astype(int)

            # Evaluate
            metrics = self._compute_metrics(y_test, y_pred, y_prob, X_test, X_train_t, y_train)

            # Log to MLflow
            mlflow.log_params({
                "n_estimators":      train_params.get("n_estimators"),
                "max_depth":         train_params.get("max_depth"),
                "learning_rate":     train_params.get("learning_rate"),
                "scale_pos_weight":  round(spw, 4),
                "n_features":        X_train_t.shape[1],
                "n_train":           len(X_train),
                "n_test":            len(X_test),
                "positive_rate_train": round(y_train.mean(), 4),
            })
            mlflow.log_metrics(metrics)

            # Log model
            if self.mlflow_cfg.get("log_artifacts"):
                mlflow.sklearn.log_model(
                    self.calibrated_model,
                    artifact_path="model",
                    registered_model_name=self.mlflow_cfg.get("model_registry_name"),
                )
                # Also save preprocessor
                mlflow.sklearn.log_model(preprocessor, artifact_path="preprocessor")

            # Save locally
            self._save_artifacts(preprocessor)

            logger.info("\n  ── Evaluation Results ──────────────────")
            for k, v in metrics.items():
                logger.info(f"    {k:35s}: {v:.4f}")

        return {
            "model":            self.model,
            "calibrated_model": self.calibrated_model,
            "preprocessor":     preprocessor,
            "feature_names":    feature_names,
            "metrics":          metrics,
            "run_id":           self.run_id,
        }

    # ── Metrics ───────────────────────────────────────────────────────────

    def _compute_metrics(
        self,
        y_test: pd.Series,
        y_pred: np.ndarray,
        y_prob: np.ndarray,
        X_test_raw: pd.DataFrame,
        X_train_t: np.ndarray,
        y_train: pd.Series,
    ) -> Dict[str, float]:
        from sklearn.metrics import (
            roc_auc_score, average_precision_score, f1_score,
            precision_score, recall_score, brier_score_loss,
        )

        metrics = {}

        # Guard: handle near-constant y_test
        if y_test.nunique() < 2:
            logger.warning("  Only one class in test set — limited metrics available")
            metrics["f1"] = float(f1_score(y_test, y_pred, zero_division=0))
            return metrics

        metrics["roc_auc"]            = float(roc_auc_score(y_test, y_prob))
        metrics["pr_auc"]             = float(average_precision_score(y_test, y_prob))
        metrics["f1"]                 = float(f1_score(y_test, y_pred, zero_division=0))
        metrics["precision"]          = float(precision_score(y_test, y_pred, zero_division=0))
        metrics["recall"]             = float(recall_score(y_test, y_pred, zero_division=0))
        metrics["brier_score"]        = float(brier_score_loss(y_test, y_prob))

        # Precision@top-20%: are the highest-risk children actually defaulters?
        n_top = max(1, int(0.20 * len(y_test)))
        top_idx = np.argsort(y_prob)[-n_top:]
        metrics["precision_at_20pct"] = float(y_test.iloc[top_idx].mean())

        # ECE (calibration)
        metrics["expected_calibration_error"] = float(self._ece(y_test, y_prob))

        # Fairness: AUC by sex (if available)
        if "patient_sex_binary" in X_test_raw.columns:
            for sex_val, label in [(1, "male"), (0, "female")]:
                mask = X_test_raw["patient_sex_binary"] == sex_val
                if mask.sum() > 2 and y_test[mask].nunique() > 1:
                    metrics[f"roc_auc_{label}"] = float(
                        roc_auc_score(y_test[mask], y_prob[mask])
                    )

        return metrics

    @staticmethod
    def _ece(y_true: pd.Series, y_prob: np.ndarray, n_bins: int = 10) -> float:
        """Expected Calibration Error."""
        bins = np.linspace(0, 1, n_bins + 1)
        ece  = 0.0
        n    = len(y_true)
        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() == 0:
                continue
            frac = mask.sum() / n
            acc  = float(y_true[mask].mean())
            conf = float(y_prob[mask].mean())
            ece += frac * abs(acc - conf)
        return ece

    # ── Artifacts ─────────────────────────────────────────────────────────

    def _save_artifacts(self, preprocessor) -> None:
        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)

        joblib.dump(self.calibrated_model, out_dir / "model.pkl")
        joblib.dump(preprocessor, out_dir / "preprocessor.pkl")
        joblib.dump(self.feature_names, out_dir / "feature_names.pkl")

        logger.info(f"  Artifacts saved to {out_dir}/")

    @staticmethod
    def load_artifacts(processed_dir: str = "data/processed"):
        """Load saved model, preprocessor, and feature names."""
        d = Path(processed_dir)
        model         = joblib.load(d / "model.pkl")
        preprocessor  = joblib.load(d / "preprocessor.pkl")
        feature_names = joblib.load(d / "feature_names.pkl")
        return model, preprocessor, feature_names
