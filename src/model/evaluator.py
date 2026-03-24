"""
model/evaluator.py
─────────────────────────────────────────────────────────────────────────────
Comprehensive evaluation: ROC/PR curves, calibration plot, fairness analysis,
feature importance. All plots saved to reports/ and logged to MLflow.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import (
    roc_curve, precision_recall_curve, auc,
    confusion_matrix, classification_report,
)

logger = logging.getLogger(__name__)

REPORTS_DIR = Path("reports")
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


class ModelEvaluator:

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)
        self.thresholds = self.cfg["evaluation"]["thresholds"]

    def full_evaluation(
        self,
        model,
        preprocessor,
        X_test: pd.DataFrame,
        y_test: pd.Series,
        feature_names: List[str],
    ) -> Dict:
        """Run all evaluation routines and return summary dict."""
        X_t    = preprocessor.transform(X_test)
        y_prob = model.predict_proba(X_t)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)

        results = {}
        results["roc_pr_plot"]     = self._plot_roc_pr(y_test, y_prob)
        results["calibration_plot"]= self._plot_calibration(y_test, y_prob)
        results["confusion_matrix"]= self._plot_confusion_matrix(y_test, y_pred)
        results["fairness"]        = self._fairness_analysis(model, preprocessor, X_test, y_test)
        results["feature_importance"] = self._feature_importance(model, feature_names)
        results["threshold_analysis"]  = self._threshold_analysis(y_test, y_prob)
        results["report_text"]     = classification_report(y_test, y_pred,
                                                           target_names=["on-track","defaulter"])

        logger.info("\n── Classification Report ────────────────────")
        logger.info(results["report_text"])
        self._check_thresholds(results)
        return results

    # ── Plots ──────────────────────────────────────────────────────────────

    def _plot_roc_pr(self, y_test, y_prob) -> str:
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        if y_test.nunique() > 1:
            # ROC
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            roc_auc = auc(fpr, tpr)
            axes[0].plot(fpr, tpr, color="#2E75B6", lw=2,
                         label=f"ROC (AUC = {roc_auc:.3f})")
            axes[0].plot([0,1],[0,1],"--", color="#AAAAAA")
            axes[0].set_xlabel("False Positive Rate")
            axes[0].set_ylabel("True Positive Rate")
            axes[0].set_title("ROC Curve")
            axes[0].legend()

            # PR
            precision, recall, _ = precision_recall_curve(y_test, y_prob)
            pr_auc = auc(recall, precision)
            axes[1].plot(recall, precision, color="#C00000", lw=2,
                         label=f"PR (AUC = {pr_auc:.3f})")
            baseline = float(y_test.mean())
            axes[1].axhline(y=baseline, color="#AAAAAA", linestyle="--",
                            label=f"Baseline ({baseline:.2f})")
            axes[1].set_xlabel("Recall")
            axes[1].set_ylabel("Precision")
            axes[1].set_title("Precision-Recall Curve")
            axes[1].legend()
        else:
            for ax in axes:
                ax.text(0.5, 0.5, "Single class in test set", ha="center", va="center")

        plt.tight_layout()
        path = str(REPORTS_DIR / "roc_pr_curves.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {path}")
        return path

    def _plot_calibration(self, y_test, y_prob, n_bins: int = 8) -> str:
        fig, ax = plt.subplots(figsize=(6, 6))

        bins       = np.linspace(0, 1, n_bins + 1)
        bin_means  = []
        actual_pos = []

        for lo, hi in zip(bins[:-1], bins[1:]):
            mask = (y_prob >= lo) & (y_prob < hi)
            if mask.sum() > 0:
                bin_means.append(float(y_prob[mask].mean()))
                actual_pos.append(float(y_test[mask].mean()))

        ax.plot([0,1],[0,1],"--", color="#AAAAAA", label="Perfect calibration")
        if bin_means:
            ax.plot(bin_means, actual_pos, "o-", color="#2E75B6", lw=2,
                    label="Model calibration")
        ax.set_xlabel("Mean Predicted Probability")
        ax.set_ylabel("Fraction of Positives")
        ax.set_title("Calibration Curve (Reliability Diagram)")
        ax.legend()
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)

        path = str(REPORTS_DIR / "calibration_curve.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {path}")
        return path

    def _plot_confusion_matrix(self, y_test, y_pred) -> str:
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(5, 4))
        im = ax.imshow(cm, cmap="Blues")
        ax.set_xticks([0,1]); ax.set_yticks([0,1])
        ax.set_xticklabels(["On-track","Defaulter"])
        ax.set_yticklabels(["On-track","Defaulter"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i,j]), ha="center", va="center",
                        color="white" if cm[i,j] > cm.max()/2 else "black", fontsize=16)
        plt.colorbar(im, ax=ax)
        path = str(REPORTS_DIR / "confusion_matrix.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    # ── Fairness ───────────────────────────────────────────────────────────

    def _fairness_analysis(
        self, model, preprocessor, X_test: pd.DataFrame, y_test: pd.Series
    ) -> pd.DataFrame:
        from sklearn.metrics import roc_auc_score, average_precision_score
        rows = []
        X_t  = preprocessor.transform(X_test)
        y_prob = model.predict_proba(X_t)[:, 1]

        subgroups = {
            "sex":      ("patient_sex_binary", {1: "male", 0: "female"}),
            "county":   ("county_encoded",     None),
        }
        for dim, (col, labels) in subgroups.items():
            if col not in X_test.columns:
                continue
            for val in X_test[col].dropna().unique():
                mask = X_test[col] == val
                if mask.sum() < 3 or y_test[mask].nunique() < 2:
                    continue
                label = labels.get(int(val), str(val)) if labels else str(val)
                try:
                    row = {
                        "dimension":  dim,
                        "group":      label,
                        "n":          int(mask.sum()),
                        "pos_rate":   round(float(y_test[mask].mean()), 3),
                        "roc_auc":    round(float(roc_auc_score(y_test[mask], y_prob[mask])), 3),
                        "pr_auc":     round(float(average_precision_score(y_test[mask], y_prob[mask])), 3),
                    }
                    rows.append(row)
                except Exception:
                    pass

        df = pd.DataFrame(rows)
        if not df.empty:
            logger.info(f"\n  Fairness analysis:\n{df.to_string(index=False)}")
        return df

    # ── Feature importance ─────────────────────────────────────────────────

    def _feature_importance(self, model, feature_names: List[str]) -> pd.DataFrame:
        """Extract XGBoost feature importance and plot."""
        # Get the underlying XGBoost model (unwrap CalibratedClassifierCV)
        xgb_model = model
        if hasattr(model, "calibrated_classifiers_"):
            xgb_model = model.calibrated_classifiers_[0].estimator
        elif hasattr(model, "estimator"):
            xgb_model = model.estimator

        if not hasattr(xgb_model, "feature_importances_"):
            logger.warning("  Cannot extract feature importance from this model type")
            return pd.DataFrame()

        importances = xgb_model.feature_importances_
        n = min(len(importances), len(feature_names))
        fi = pd.DataFrame({
            "feature":    feature_names[:n],
            "importance": importances[:n],
        }).sort_values("importance", ascending=False)

        # Plot top 20
        top = fi.head(20)
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.barh(top["feature"][::-1], top["importance"][::-1], color="#2E75B6")
        ax.set_xlabel("XGBoost Feature Importance (gain)")
        ax.set_title("Top 20 Feature Importances")
        plt.tight_layout()
        path = str(REPORTS_DIR / "feature_importance.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Saved: {path}")
        return fi

    # ── Threshold analysis ─────────────────────────────────────────────────

    def _threshold_analysis(self, y_test, y_prob) -> pd.DataFrame:
        """Precision/recall trade-off at multiple thresholds."""
        rows = []
        for t in np.arange(0.1, 0.9, 0.05):
            yp = (y_prob >= t).astype(int)
            tp = int(((yp == 1) & (y_test == 1)).sum())
            fp = int(((yp == 1) & (y_test == 0)).sum())
            fn = int(((yp == 0) & (y_test == 1)).sum())
            prec = tp / max(tp + fp, 1)
            rec  = tp / max(tp + fn, 1)
            rows.append({
                "threshold": round(float(t), 2),
                "flagged":   int(yp.sum()),
                "precision": round(prec, 3),
                "recall":    round(rec, 3),
                "f1":        round(2*prec*rec / max(prec+rec, 0.001), 3),
            })
        df = pd.DataFrame(rows)
        logger.info(f"\n  Threshold analysis:\n{df.to_string(index=False)}")
        return df

    # ── Threshold checks ───────────────────────────────────────────────────

    def _check_thresholds(self, results: Dict):
        logger.info("\n── Deployment Readiness Checks ─────────────")
        checks = {
            "PR-AUC ≥ threshold":          ("pr_auc", self.thresholds["min_pr_auc"]),
        }
        # These come from the trainer metrics, not stored here — warn only
        logger.info("  (Threshold checks run during trainer.train() — see MLflow metrics)")
