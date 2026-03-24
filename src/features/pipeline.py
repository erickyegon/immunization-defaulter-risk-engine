"""
features/pipeline.py
─────────────────────────────────────────────────────────────────────────────
Selects and preprocesses the final feature matrix for XGBoost training.
Handles: column selection, imputation, encoding, and dtype enforcement.
"""

import logging
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

logger = logging.getLogger(__name__)

# ── Feature column definitions (aligned with merger output) ───────────────

FEATURE_GROUPS = {
    "child_numeric": [
        "patient_age_in_months",
        "vax_completeness_score",
        "vax_completeness_all",
        "age_expected_vaccine_count",
        "due_count_clean",
        "vitamin_a_completeness",
        "months_since_reported",
        "vax_received_core",
        "vax_received_total",
    ],
    "child_binary": [
        "patient_sex_binary",
        "is_malaria_endemic_binary",
        "is_growth_monitoring_binary",
        "has_delayed_milestones_binary",
        "penta_series_complete",
        "opv_series_complete",
        "pcv_series_complete",
        "rota_series_complete",
        "measles_booster_gap",
        # Individual vaccine flags (for SHAP)
        "has_bcg", "has_opv_0", "has_opv_1", "has_opv_2", "has_opv_3",
        "has_pcv_1", "has_pcv_2", "has_pcv_3",
        "has_penta_1", "has_penta_2", "has_penta_3",
        "has_ipv", "has_rota_1", "has_rota_2", "has_rota_3",
        "has_measles_9_months", "has_measles_18_months",
    ],
    "chw_numeric": [
        "chw_supervision_frequency",
        "chw_immunization_competency_pct",
        "chw_overall_assessment_pct",
        "chw_workload_u2",
        "monthly_homevisit_rate",
        "months_since_last_supervision",
    ],
    "chw_binary": [
        "chw_has_all_tools",
        "chw_has_ppe",
    ],
    "maternal_numeric": [
        "maternal_anc_visits",
    ],
    "maternal_binary": [
        "maternal_anc_defaulter",
        "maternal_muac_risk",
        "maternal_iron_folate",
    ],
    "engagement_binary": [
        "household_on_fp",
    ],
    "geographic_categorical": [
        "sub_county_encoded",
        "county_encoded",
    ],
}

TARGET_COL = "is_defaulter"


class FeaturePipeline:
    """
    Selects features from the analytical dataset and builds the
    sklearn-compatible preprocessor for the XGBoost pipeline.
    """

    def __init__(self, config_path: str = "config/model_config.yaml"):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

    def select_features(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Select available features from analytical DataFrame.
        Returns (X, y) — missing columns filled with NaN (not dropped).
        """
        all_features = self._all_feature_cols()

        # Only keep features that exist in df
        available = [c for c in all_features if c in df.columns]
        missing   = [c for c in all_features if c not in df.columns]

        if missing:
            logger.info(f"  Features not yet available (join not resolved): {missing[:10]}{'...' if len(missing)>10 else ''}")
            for col in missing:
                df[col] = np.nan

        X = df[all_features].copy()
        y = df[TARGET_COL].copy()

        logger.info(f"  Feature matrix: {X.shape[0]} samples × {X.shape[1]} features")
        logger.info(f"  Target: {y.sum()} positives ({y.mean():.1%})")
        return X, y

    def build_preprocessor(self, X: pd.DataFrame) -> ColumnTransformer:
        """
        Build sklearn ColumnTransformer with appropriate imputers.
        Returns unfitted preprocessor (fit during training).
        All-null columns are excluded — they carry no information and cause
        sklearn imputer warnings.
        """
        numeric_cols = [c for c in self._numeric_cols(X) if X[c].notna().any()]
        binary_cols  = [c for c in self._binary_cols(X)  if X[c].notna().any()]
        cat_cols     = [c for c in self._categorical_cols(X) if X[c].notna().any()]

        all_null = [
            c for c in self._numeric_cols(X) + self._binary_cols(X) + self._categorical_cols(X)
            if not X[c].notna().any()
        ]
        if all_null:
            logger.warning(
                f"  Dropping {len(all_null)} all-null feature(s) from preprocessor "
                f"(no data available): {all_null}"
            )

        transformers = []

        if numeric_cols:
            transformers.append((
                "numeric",
                SimpleImputer(strategy="median"),
                numeric_cols,
            ))

        if binary_cols:
            transformers.append((
                "binary",
                SimpleImputer(strategy="most_frequent"),
                binary_cols,
            ))

        if cat_cols:
            transformers.append((
                "categorical",
                Pipeline([
                    ("impute", SimpleImputer(strategy="most_frequent")),
                    ("encode", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
                ]),
                cat_cols,
            ))

        preprocessor = ColumnTransformer(
            transformers=transformers,
            remainder="drop",
            verbose_feature_names_out=False,
        )
        logger.info(f"  Preprocessor: {len(numeric_cols)} numeric, "
                    f"{len(binary_cols)} binary, {len(cat_cols)} categorical")
        return preprocessor

    def get_feature_names(self, preprocessor: ColumnTransformer, X: pd.DataFrame) -> List[str]:
        """Return feature names in preprocessor output order."""
        names = []
        for name, trans, cols in preprocessor.transformers_:
            if name == "remainder":
                continue
            names.extend(cols)
        return names

    def report(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """Per-feature null rate and variance report."""
        rows = []
        for col in X.columns:
            rows.append({
                "feature":    col,
                "null_pct":   round(X[col].isna().mean() * 100, 1),
                "unique":     X[col].nunique(),
                "mean":       round(X[col].mean(), 4) if pd.api.types.is_numeric_dtype(X[col]) else None,
                "std":        round(X[col].std(), 4)  if pd.api.types.is_numeric_dtype(X[col]) else None,
            })
        return pd.DataFrame(rows).sort_values("null_pct", ascending=False)

    # ── Helpers ────────────────────────────────────────────────────────────

    def _all_feature_cols(self) -> List[str]:
        cols = []
        for group in FEATURE_GROUPS.values():
            cols.extend(group)
        return list(dict.fromkeys(cols))  # deduplicate preserving order

    def _numeric_cols(self, X: pd.DataFrame) -> List[str]:
        candidates = (
            FEATURE_GROUPS["child_numeric"]
            + FEATURE_GROUPS["chw_numeric"]
            + FEATURE_GROUPS["maternal_numeric"]
        )
        return [c for c in candidates if c in X.columns]

    def _binary_cols(self, X: pd.DataFrame) -> List[str]:
        candidates = (
            FEATURE_GROUPS["child_binary"]
            + FEATURE_GROUPS["chw_binary"]
            + FEATURE_GROUPS["maternal_binary"]
            + FEATURE_GROUPS["engagement_binary"]
        )
        return [c for c in candidates if c in X.columns]

    def _categorical_cols(self, X: pd.DataFrame) -> List[str]:
        return [c for c in FEATURE_GROUPS["geographic_categorical"] if c in X.columns]


# ── Vitamin A completeness (called during merge phase) ────────────────────

def compute_vitamin_a_completeness(df: pd.DataFrame) -> pd.Series:
    """
    Compute proportion of expected Vitamin A doses received.
    Requires patient_age_in_months and has_vitamin_a_* columns.
    """
    from config.epi_schedule import VITAMIN_A_SCHEDULE, get_expected_vitamin_a

    received = pd.Series(0.0, index=df.index)
    age = pd.to_numeric(df["patient_age_in_months"], errors="coerce")
    expected = age.apply(lambda a: get_expected_vitamin_a(a) if pd.notna(a) else 0)

    for col, min_age in VITAMIN_A_SCHEDULE:
        if col not in df.columns:
            continue
        eligible = df["patient_age_in_months"] >= min_age
        received += (
            pd.to_numeric(df[col], errors="coerce").fillna(0) * eligible.astype(float)
        )

    completeness = np.where(expected > 0, received / expected, np.nan)
    return pd.Series(np.clip(completeness, 0, 1), index=df.index)
