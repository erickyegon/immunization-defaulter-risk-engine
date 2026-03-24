"""
explainability/shap_explainer.py
─────────────────────────────────────────────────────────────────────────────
Global and per-patient SHAP explanations using TreeExplainer.
Outputs:
  - Global beeswarm + bar plots (reports/shap/)
  - Per-patient JSON explanation (API-ready)
  - Waterfall plots for individual patients
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap
import yaml

logger = logging.getLogger(__name__)


class SHAPExplainer:
    """
    Wraps shap.TreeExplainer for both global analysis and per-patient predictions.
    """

    # Human-readable feature name mapping for CHW-facing explanations
    FEATURE_LABELS = {
        "vax_completeness_score":            "Overall vaccine completeness",
        "patient_age_in_months":             "Child's age (months)",
        "due_count_clean":                   "Doses currently outstanding",
        "months_since_reported":             "Months since last CHW contact",
        "chw_supervision_frequency":         "CHW supervision visits (count)",
        "chw_immunization_competency_pct":   "CHW immunization knowledge score",
        "chw_workload_u2":                   "Under-2 children per CHW area",
        "monthly_homevisit_rate":            "Monthly home visit rate in area",
        "maternal_anc_visits":               "Mother's ANC visits attended",
        "maternal_anc_defaulter":            "Mother defaulted on ANC",
        "maternal_muac_risk":                "Mother had nutritional risk (MUAC)",
        "measles_booster_gap":               "Missed measles booster (18-month)",
        "penta_series_complete":             "Penta 1-2-3 series complete",
        "has_penta_1":                       "Penta-1 received",
        "has_penta_2":                       "Penta-2 received",
        "has_penta_3":                       "Penta-3 received",
        "has_measles_9_months":              "Measles-Rubella MR1 received",
        "has_measles_18_months":             "Measles-Rubella MR2 received",
        "patient_sex_binary":               "Child's sex (male)",
        "is_growth_monitoring_binary":       "Enrolled in growth monitoring",
        "chw_has_all_tools":                 "CHW has all required tools",
        "household_on_fp":                   "Household using family planning",
        "sub_county_encoded":                "Sub-county",
    }

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            cfg = yaml.safe_load(f)
        self.shap_cfg = cfg["shap"]
        self.api_cfg  = cfg["api"]
        self.out_dir  = Path(self.shap_cfg["output_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)

        self.explainer     = None
        self.shap_values   = None
        self.feature_names: List[str] = []

    # ── Setup ──────────────────────────────────────────────────────────────

    def fit(self, model, X_transformed: np.ndarray, feature_names: List[str]) -> None:
        """
        Fit TreeExplainer on the XGBoost model.
        Must be called before global_analysis() or explain_patient().
        """
        # Unwrap CalibratedClassifierCV to get raw XGBoost
        xgb_model = self._unwrap_model(model)
        self.feature_names = feature_names

        logger.info("  Fitting SHAP TreeExplainer...")
        self.explainer = shap.TreeExplainer(
            xgb_model,
            feature_perturbation="interventional",
        )

        # Compute SHAP values on background sample
        n_bg = min(self.shap_cfg["n_background_samples"], X_transformed.shape[0])
        bg   = X_transformed[:n_bg]
        self.shap_values = self.explainer.shap_values(bg)
        logger.info(f"  SHAP values computed: {self.shap_values.shape}")

    # ── Global Analysis ────────────────────────────────────────────────────

    def global_analysis(self, X_transformed: np.ndarray) -> Dict:
        """
        Generate global SHAP plots: beeswarm + bar chart.
        Returns dict of {plot_name: file_path}.
        """
        if self.explainer is None:
            raise RuntimeError("Call .fit() before global_analysis()")

        shap_vals = self.explainer.shap_values(X_transformed)
        paths = {}

        # ── Beeswarm plot ──────────────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.summary_plot(
            shap_vals,
            X_transformed,
            feature_names=self._friendly_names(),
            max_display=self.shap_cfg["max_display"],
            show=False,
            plot_type="dot",
        )
        plt.title("SHAP Feature Impact on Immunization Default Risk", fontsize=13, pad=15)
        plt.tight_layout()
        p = str(self.out_dir / "shap_beeswarm.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["beeswarm"] = p
        logger.info(f"  Saved: {p}")

        # ── Bar chart (mean |SHAP|) ────────────────────────────────────────
        fig, ax = plt.subplots(figsize=(9, 7))
        shap.summary_plot(
            shap_vals,
            X_transformed,
            feature_names=self._friendly_names(),
            max_display=self.shap_cfg["max_display"],
            show=False,
            plot_type="bar",
        )
        plt.title("Mean |SHAP Value| — Global Feature Importance", fontsize=13)
        plt.tight_layout()
        p = str(self.out_dir / "shap_bar.png")
        plt.savefig(p, dpi=150, bbox_inches="tight")
        plt.close()
        paths["bar"] = p
        logger.info(f"  Saved: {p}")

        # Mean importance table (guard against length mismatch)
        mean_shap = np.abs(shap_vals).mean(axis=0)
        n = min(len(self.feature_names), len(mean_shap))
        fn = self.feature_names[:n]
        importance_df = pd.DataFrame({
            "feature":      fn,
            "friendly_name":[self.FEATURE_LABELS.get(f, f) for f in fn],
            "mean_shap":    mean_shap[:n],
        }).sort_values("mean_shap", ascending=False)

        importance_df.to_csv(str(self.out_dir / "shap_importance.csv"), index=False)
        paths["importance_csv"] = str(self.out_dir / "shap_importance.csv")

        logger.info("\n  Top 10 features by mean |SHAP|:")
        for _, row in importance_df.head(10).iterrows():
            logger.info(f"    {row['friendly_name']:45s} {row['mean_shap']:.4f}")

        return paths

    # ── Per-Patient Explanation ────────────────────────────────────────────

    def explain_patient(
        self,
        patient_row: pd.Series,
        preprocessor,
        model,
        patient_meta: Optional[Dict] = None,
    ) -> Dict:
        """
        Generate a structured per-patient SHAP explanation.
        Returns the API response JSON dict.
        """
        if self.explainer is None:
            raise RuntimeError("Call .fit() before explain_patient()")

        X_raw = patient_row.to_frame().T
        X_t   = preprocessor.transform(X_raw)

        # Risk score
        risk_score = float(model.predict_proba(X_t)[0, 1])
        risk_tier  = self._risk_tier(risk_score)

        # SHAP values for this individual
        sv = self.explainer.shap_values(X_t)[0]

        # Top drivers
        top_n = self.api_cfg["top_shap_drivers"]
        top_idx = np.argsort(np.abs(sv))[::-1][:top_n]

        drivers = []
        for i in top_idx:
            feat_name = self.feature_names[i] if i < len(self.feature_names) else f"feature_{i}"
            feat_val  = float(X_t[0, i])
            shap_val  = float(sv[i])
            drivers.append({
                "feature":      feat_name,
                "friendly_name":self.FEATURE_LABELS.get(feat_name, feat_name),
                "feature_value":round(feat_val, 4),
                "shap_value":   round(shap_val, 4),
                "direction":    "increases_risk" if shap_val > 0 else "decreases_risk",
                "plain_english":self._plain_english(feat_name, feat_val, shap_val),
            })

        payload = {
            "patient_id":       patient_meta.get("patient_id", "unknown") if patient_meta else "unknown",
            "patient_name":     patient_meta.get("patient_name", "") if patient_meta else "",
            "risk_score":       round(risk_score, 4),
            "risk_pct":         round(risk_score * 100, 1),
            "risk_tier":        risk_tier,
            "top_drivers":      drivers,
            "recommended_action": self._recommend(risk_tier, drivers),
            "model_version":    "xgb_v1.0_iz_defaulter",
        }
        return payload

    def waterfall_plot(
        self,
        patient_row: pd.Series,
        preprocessor,
        patient_id: str = "patient",
    ) -> str:
        """Generate and save a SHAP waterfall plot for one patient."""
        X_t = preprocessor.transform(patient_row.to_frame().T)
        sv  = self.explainer.shap_values(X_t)[0]
        ev  = self.explainer.expected_value

        shap_exp = shap.Explanation(
            values        = sv,
            base_values   = ev,
            data          = X_t[0],
            feature_names = self._friendly_names(),
        )

        fig, _ = plt.subplots(figsize=(10, 6))
        shap.waterfall_plot(shap_exp, max_display=15, show=False)
        plt.title(f"Risk Explanation — Patient {patient_id}", fontsize=12)
        plt.tight_layout()
        path = str(self.out_dir / f"waterfall_{patient_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"  Waterfall saved: {path}")
        return path

    # ── Helpers ────────────────────────────────────────────────────────────

    def _friendly_names(self) -> List[str]:
        return [self.FEATURE_LABELS.get(f, f) for f in self.feature_names]

    def _risk_tier(self, score: float) -> str:
        tiers = self.api_cfg["risk_tiers"]
        if score < tiers["low"][1]:
            return "LOW"
        elif score < tiers["medium"][1]:
            return "MEDIUM"
        return "HIGH"

    def _plain_english(self, feature: str, value: float, shap_val: float) -> str:
        """Convert a (feature, value, shap) triple to CHW-readable text."""
        direction = "increases" if shap_val > 0 else "reduces"

        templates = {
            "vax_completeness_score": (
                f"Child has received {value*100:.0f}% of expected vaccines "
                f"— this {direction} defaulter risk"
            ),
            "patient_age_in_months": (
                f"Child is {value:.0f} months old "
                f"— age context {direction} defaulter risk"
            ),
            "due_count_clean": (
                f"Child has {int(value)} outstanding vaccine dose(s) "
                f"— this {direction} defaulter risk"
            ),
            "months_since_reported": (
                f"Last CHW contact was {value:.1f} month(s) ago "
                f"— this {direction} defaulter risk"
            ),
            "chw_supervision_frequency": (
                f"This CHW area had {int(value)} supervision visit(s) "
                f"— this {direction} defaulter risk"
            ),
            "maternal_anc_visits": (
                f"Mother attended {int(value)} ANC visit(s) "
                f"— this {direction} defaulter risk"
            ),
            "measles_booster_gap": (
                ("Child received MR1 but is missing the 18-month MR2 booster "
                 f"— this {direction} defaulter risk") if value == 1
                else f"Measles booster gap not detected"
            ),
        }
        return templates.get(
            feature,
            f"{self.FEATURE_LABELS.get(feature, feature)} value={value:.2f} "
            f"{direction} defaulter risk"
        )

    def _recommend(self, tier: str, drivers: List[Dict]) -> str:
        """Generate CHW action recommendation based on risk tier and drivers."""
        base = {
            "LOW":    "Routine follow-up at next scheduled visit.",
            "MEDIUM": "Prioritise within-month home visit. Review vaccine card.",
            "HIGH":   "Immediate home visit required. Bring vaccine referral form.",
        }[tier]

        # Add driver-specific detail for HIGH risk
        if tier == "HIGH" and drivers:
            top_feat = drivers[0]["feature"]
            extras = {
                "measles_booster_gap":
                    " Specifically: schedule MR2 booster referral.",
                "due_count_clean":
                    " Check outstanding doses and arrange facility referral.",
                "months_since_reported":
                    " Child has not been seen recently — locate household.",
            }
            base += extras.get(top_feat, "")
        return base

    @staticmethod
    def _unwrap_model(model):
        """Extract base XGBoost from CalibratedClassifierCV wrapper."""
        if hasattr(model, "calibrated_classifiers_"):
            return model.calibrated_classifiers_[0].estimator
        if hasattr(model, "estimator"):
            return model.estimator
        return model
