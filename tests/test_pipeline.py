"""
tests/test_pipeline.py
─────────────────────────────────────────────────────────────────────────────
Unit tests covering ETL, feature engineering, model training, SHAP,
and API. Runs on synthetic mini-datasets — no real patient data required.

Usage:
  cd iz_defaulter_model
  python -m pytest tests/ -v
"""

import json
import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))


# ══ Fixtures ══════════════════════════════════════════════════════════════════

@pytest.fixture
def synthetic_iz():
    """Minimal iz table matching the real schema."""
    np.random.seed(42)
    n = 60
    ages = np.random.randint(0, 60, n)
    return pd.DataFrame({
        "patient_id":                  [f"pid_{i}" for i in range(n)],
        "patient_name":                [f"Child {i}" for i in range(n)],
        "patient_age_in_months":       ages,
        "patient_age_in_weeks":        ages * 4,
        "patient_age_in_days":         ages * 30,
        "patient_sex":                 np.random.choice(["male", "female"], n),
        "contact_id":                  [f"cid_{i}" for i in range(n)],
        "contact_parent_id":           np.random.choice([f"area_{j}" for j in range(5)], n),
        "contact_parent_parent_id":    [f"hh_{i%20}" for i in range(n)],
        "county":                      np.random.choice(["Busia", "Old"], n),
        "month":                       np.random.choice(["Feb25_2", "Mar25", "Dec24"], n),
        "reported":                    pd.date_range("2025-01-01", periods=n, freq="D").astype(str),
        "needs_follow_up":             np.random.choice(["yes", "no"], n, p=[0.25, 0.75]),
        "needs_immunization_follow_up":np.random.choice(["yes", "no"], n, p=[0.15, 0.85]),
        "has_good_immunization_status":["no"] * n,
        "due_count":                   np.random.choice([-1, 0, 1, 2, 3], n),
        "total_due_vaccines":          np.random.randint(0, 18, n),
        "vaccines_due_missed":         np.random.choice([None, "measles_9_months", "penta_3"], n),
        "has_bcg":                     np.random.randint(0, 2, n),
        "has_opv_0":                   np.random.randint(0, 2, n),
        "has_opv_1":                   np.random.randint(0, 2, n),
        "has_opv_2":                   np.random.randint(0, 2, n),
        "has_opv_3":                   np.random.randint(0, 2, n),
        "has_pcv_1":                   np.random.randint(0, 2, n),
        "has_pcv_2":                   np.random.randint(0, 2, n),
        "has_pcv_3":                   np.random.randint(0, 2, n),
        "has_penta_1":                 np.random.randint(0, 2, n),
        "has_penta_2":                 np.random.randint(0, 2, n),
        "has_penta_3":                 np.random.randint(0, 2, n),
        "has_ipv":                     np.random.randint(0, 2, n),
        "has_rota_1":                  np.random.randint(0, 2, n),
        "has_rota_2":                  np.random.randint(0, 2, n),
        "has_rota_3":                  np.random.randint(0, 2, n),
        "has_measles_9_months":        np.where(ages >= 9, np.random.randint(0, 2, n), 0),
        "has_measles_18_months":       np.where(ages >= 18, np.random.randint(0, 2, n), 0),
        "has_malaria_6_vaccine":       np.random.choice([0, 1, np.nan], n),
        "has_malaria_7_vaccine":       np.random.choice([0, 1, np.nan], n),
        "has_malaria_8_vaccine":       np.random.choice([0, 1, np.nan], n),
        "has_malaria_24_vaccine":      np.random.choice([0, 1, np.nan], n),
        "has_vitamin_a_6_months":      np.where(ages >= 6,  np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_12_months":     np.where(ages >= 12, np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_18_months":     np.where(ages >= 18, np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_24_months":     np.where(ages >= 24, np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_30_months":     np.where(ages >= 30, np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_36_months":     np.where(ages >= 36, np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_42_months":     np.where(ages >= 42, np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_48_months":     np.where(ages >= 48, np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_54_months":     np.where(ages >= 54, np.random.randint(0, 2, n), 0).astype(float),
        "has_vitamin_a_60_months":     np.where(ages >= 60, np.random.randint(0, 2, n), 0).astype(float),
        "is_in_malaria_endemic_region":np.random.choice(["yes", "no"], n),
        "is_available":                np.random.choice(["yes", "no"], n),
        "is_participating_in_monthly_growth_monitoring": np.random.choice(["yes", "no"], n),
        "has_signs_of_delayed_milestones": np.random.choice(["yes", "no", None], n),
        "is_participating_in_growth_monitoring": np.random.choice(["yes", "no"], n),
        "latitude":                    np.random.uniform(-1, 1, n),
        "longitude":                   np.random.uniform(33, 35, n),
        "reportedm":                   np.random.choice(["Feb25", "Mar25"], n),
        "record_hash":                 [f"hash_{i}" for i in range(n)],
        "_source_table":               ["iz"] * n,
    })


@pytest.fixture
def synthetic_supervision():
    n = 10
    return pd.DataFrame({
        "chw_area":              [f"area_{j}" for j in range(5)] * 2,
        "chw_uuid":              [f"chw_{j}" for j in range(5)] * 2,
        "reported":              pd.date_range("2025-01-01", periods=n, freq="15D").astype(str),
        "calc_immunization_score":       np.random.uniform(0, 5, n),
        "calc_immunization_denominator": np.full(n, 5.0),
        "calc_assessment_score":         np.random.uniform(0, 10, n),
        "calc_assessment_denominator":   np.full(n, 10.0),
        "pct_immunization_pct":          np.random.uniform(0, 1, n),
        "pct_assessment_pct":            np.random.uniform(0, 1, n),
        "has_all_tools":                 np.random.choice(["yes", "no"], n),
        "has_all_tools_binary":          np.random.randint(0, 2, n),
        "has_proper_protective_equipment": np.random.choice(["yes", "no"], n),
        "has_proper_protective_equipment_binary": np.random.randint(0, 2, n),
        "_source_table": ["supervision"] * n,
    })


@pytest.fixture
def synthetic_population():
    return pd.DataFrame({
        "chw_area":   [f"area_{j}" for j in range(5)],
        "chw_uuid":   [f"chw_{j}" for j in range(5)],
        "u2_pop":     [20, 25, 30, 15, 35],
        "reportedm":  ["Feb25"] * 5,
        "_source_table": ["population"] * 5,
    })


@pytest.fixture
def synthetic_homevisit():
    n = 30
    return pd.DataFrame({
        "chw_area":   np.random.choice([f"area_{j}" for j in range(5)], n),
        "chw_uuid":   [f"chw_{i%5}" for i in range(n)],
        "family_id":  [f"fam_{i}" for i in range(n)],
        "reported":   pd.date_range("2025-01-01", periods=n, freq="3D").astype(str),
        "_source_table": ["homevisit"] * n,
    })


# ══ EPI Schedule Tests ═══════════════════════════════════════════════════════

class TestEPISchedule:
    def test_newborn_expects_2_vaccines(self):
        from config.epi_schedule import get_expected_vaccines
        assert get_expected_vaccines(0) == 2  # BCG + OPV-0

    def test_6_week_child_expects_8_vaccines(self):
        from config.epi_schedule import get_expected_vaccines
        # BCG, OPV-0, OPV-1, PCV-1, Penta-1, Rota-1 = 6 core
        result = get_expected_vaccines(2)  # ~8 weeks
        assert result >= 6

    def test_9_month_child_includes_measles(self):
        from config.epi_schedule import get_expected_vaccines
        result = get_expected_vaccines(9)
        assert result >= 15  # includes MR1

    def test_malaria_endemic_adds_doses(self):
        from config.epi_schedule import get_expected_vaccines
        core   = get_expected_vaccines(8, malaria_endemic=False)
        endemic = get_expected_vaccines(8, malaria_endemic=True)
        assert endemic > core

    def test_nan_age_returns_zero(self):
        from config.epi_schedule import get_expected_vaccines
        assert get_expected_vaccines(float("nan")) == 0

    def test_vitamin_a_schedule(self):
        from config.epi_schedule import get_expected_vitamin_a
        assert get_expected_vitamin_a(5)  == 0   # not yet due
        assert get_expected_vitamin_a(6)  == 1
        assert get_expected_vitamin_a(12) == 2
        assert get_expected_vitamin_a(60) == 10


# ══ Cleaner Tests ═════════════════════════════════════════════════════════════

class TestDataCleaner:
    def test_month_standardisation(self):
        from src.etl.cleaner import DataCleaner
        cleaner = DataCleaner()
        assert cleaner._parse_month("Feb25_2") == "2025-02"
        assert cleaner._parse_month("Dec24")   == "2024-12"
        assert cleaner._parse_month("Mar25")   == "2025-03"

    def test_sentinel_due_count_recoded(self, synthetic_iz):
        from src.etl.cleaner import DataCleaner
        cleaner = DataCleaner()
        df = cleaner.clean_iz(synthetic_iz.copy())
        assert (df["due_count_clean"] >= 0).all() or df["due_count_clean"].isna().any()
        assert -1 not in df["due_count_clean"].dropna().values

    def test_sex_binary_encoding(self, synthetic_iz):
        from src.etl.cleaner import DataCleaner
        cleaner = DataCleaner()
        df = cleaner.clean_iz(synthetic_iz.copy())
        assert set(df["patient_sex_binary"].dropna().unique()).issubset({0, 1})

    def test_malaria_nulls_recoded_for_nonendemics(self, synthetic_iz):
        from src.etl.cleaner import DataCleaner
        synthetic_iz["is_in_malaria_endemic_region"] = "no"
        synthetic_iz["has_malaria_6_vaccine"]        = np.nan
        cleaner = DataCleaner()
        df = cleaner.clean_iz(synthetic_iz.copy())
        assert df["has_malaria_6_vaccine"].notna().all()
        assert (df["has_malaria_6_vaccine"] == 0).all()

    def test_supervision_infinity_fixed(self, synthetic_supervision):
        from src.etl.cleaner import DataCleaner
        synthetic_supervision["perce_campaign_service_score"] = np.inf
        cleaner = DataCleaner()
        df = cleaner.clean_supervision(synthetic_supervision.copy())
        assert not np.isinf(df.select_dtypes(include=np.number).values).any()

    def test_json_parse_safe(self):
        from src.etl.cleaner import DataCleaner
        assert DataCleaner._parse_json_safe(None)   == {}
        assert DataCleaner._parse_json_safe("bad")  == {}
        assert DataCleaner._parse_json_safe('{"k":"v"}') == {"k": "v"}


# ══ Feature Pipeline Tests ════════════════════════════════════════════════════

class TestFeaturePipeline:
    def test_vaccine_completeness_bounds(self, synthetic_iz):
        from src.etl.cleaner import DataCleaner
        from src.etl.merger  import DataMerger

        cleaner = DataCleaner()
        iz_clean = cleaner.clean_iz(synthetic_iz.copy())
        merger   = DataMerger()
        df_step2 = merger._step2_vaccine_completeness(iz_clean)

        scores = df_step2["vax_completeness_score"].dropna()
        assert (scores >= 0).all(), "Score below 0"
        assert (scores <= 1).all(), "Score above 1"

    def test_target_construction(self, synthetic_iz):
        from src.etl.cleaner import DataCleaner
        from src.etl.merger  import DataMerger

        cleaner  = DataCleaner()
        iz_clean = cleaner.clean_iz(synthetic_iz.copy())
        merger   = DataMerger()
        df       = merger._step2_vaccine_completeness(iz_clean)
        df       = merger._step3_build_target(df)

        assert "is_defaulter" in df.columns
        assert set(df["is_defaulter"].unique()).issubset({0, 1})
        assert df["is_defaulter"].mean() > 0, "Zero positives — target construction failed"

    def test_feature_selection_returns_correct_shape(self, synthetic_iz):
        from src.etl.cleaner      import DataCleaner
        from src.etl.merger       import DataMerger
        from src.features.pipeline import FeaturePipeline
        from src.features.pipeline import compute_vitamin_a_completeness

        cleaner  = DataCleaner()
        iz_clean = cleaner.clean_iz(synthetic_iz.copy())
        iz_clean["vitamin_a_completeness"] = compute_vitamin_a_completeness(iz_clean)
        iz_clean["is_malaria_endemic_binary"] = (
            iz_clean.get("is_in_malaria_endemic_region", pd.Series("no"))
            .str.lower().map({"yes": 1, "no": 0})
        )
        merger = DataMerger()
        df     = merger._step2_vaccine_completeness(iz_clean)
        df     = merger._step3_build_target(df)
        df["county_encoded"]     = 0
        df["sub_county_encoded"] = 0

        fp   = FeaturePipeline("config/model_config.yaml")
        X, y = fp.select_features(df)

        assert X.shape[0] == len(df)
        assert y.shape[0] == len(df)
        assert "is_defaulter" not in X.columns

    def test_preprocessor_no_nans_after_transform(self, synthetic_iz):
        from src.etl.cleaner      import DataCleaner
        from src.etl.merger       import DataMerger
        from src.features.pipeline import FeaturePipeline, compute_vitamin_a_completeness

        cleaner  = DataCleaner()
        iz_clean = cleaner.clean_iz(synthetic_iz.copy())
        iz_clean["vitamin_a_completeness"] = compute_vitamin_a_completeness(iz_clean)
        iz_clean["is_malaria_endemic_binary"] = 0
        merger   = DataMerger()
        df       = merger._step2_vaccine_completeness(iz_clean)
        df       = merger._step3_build_target(df)
        df["county_encoded"]     = 0
        df["sub_county_encoded"] = 0

        fp           = FeaturePipeline("config/model_config.yaml")
        X, _         = fp.select_features(df)
        preprocessor = fp.build_preprocessor(X)
        X_t          = preprocessor.fit_transform(X)

        assert not np.isnan(X_t).any(), "NaNs remain after preprocessing"
        assert not np.isinf(X_t).any(), "Infs remain after preprocessing"


# ══ Model Training Tests ══════════════════════════════════════════════════════

class TestModelTrainer:
    def _build_training_data(self, synthetic_iz):
        """Shared helper to build X, y from synthetic data."""
        from src.etl.cleaner       import DataCleaner
        from src.etl.merger        import DataMerger
        from src.features.pipeline import FeaturePipeline, compute_vitamin_a_completeness

        cleaner  = DataCleaner()
        iz_clean = cleaner.clean_iz(synthetic_iz.copy())
        iz_clean["vitamin_a_completeness"]    = compute_vitamin_a_completeness(iz_clean)
        iz_clean["is_malaria_endemic_binary"] = 0
        merger   = DataMerger()
        df       = merger._step2_vaccine_completeness(iz_clean)
        df       = merger._step3_build_target(df)
        df["county_encoded"]     = 0
        df["sub_county_encoded"] = 0

        fp   = FeaturePipeline("config/model_config.yaml")
        X, y = fp.select_features(df)
        preprocessor = fp.build_preprocessor(X)
        feat_names   = fp._all_feature_cols()
        return X, y, preprocessor, feat_names

    def test_trainer_returns_required_keys(self, synthetic_iz):
        from src.model.trainer import IZDefaulterTrainer
        X, y, pre, feat = self._build_training_data(synthetic_iz)

        trainer = IZDefaulterTrainer("config/model_config.yaml")
        # Disable tuning and calibration for speed in unit tests
        trainer.calibration_cfg["enabled"] = False
        result  = trainer.train(X, y, pre, feat, run_name="unit_test")

        required = {"model", "calibrated_model", "preprocessor",
                    "feature_names", "metrics", "run_id"}
        assert required.issubset(result.keys())

    def test_model_predicts_probabilities(self, synthetic_iz):
        from src.model.trainer import IZDefaulterTrainer
        X, y, pre, feat = self._build_training_data(synthetic_iz)

        trainer = IZDefaulterTrainer("config/model_config.yaml")
        trainer.calibration_cfg["enabled"] = False
        result  = trainer.train(X, y, pre, feat, run_name="unit_test_prob")

        X_t    = result["preprocessor"].transform(X)
        y_prob = result["model"].predict_proba(X_t)[:, 1]
        assert y_prob.min() >= 0.0
        assert y_prob.max() <= 1.0

    def test_ece_bounded(self):
        from src.model.trainer import IZDefaulterTrainer
        y_true = pd.Series([1, 0, 1, 0, 1, 0])
        y_prob = np.array([0.8, 0.2, 0.7, 0.3, 0.9, 0.1])
        ece    = IZDefaulterTrainer._ece(y_true, y_prob)
        assert 0.0 <= ece <= 1.0


# ══ SHAP Tests ════════════════════════════════════════════════════════════════

class TestSHAPExplainer:
    def test_explain_patient_returns_required_keys(self, synthetic_iz):
        from src.etl.cleaner              import DataCleaner
        from src.etl.merger               import DataMerger
        from src.features.pipeline        import FeaturePipeline, compute_vitamin_a_completeness
        from src.model.trainer            import IZDefaulterTrainer
        from src.explainability.shap_explainer import SHAPExplainer

        cleaner  = DataCleaner()
        iz_clean = cleaner.clean_iz(synthetic_iz.copy())
        iz_clean["vitamin_a_completeness"]    = compute_vitamin_a_completeness(iz_clean)
        iz_clean["is_malaria_endemic_binary"] = 0
        merger   = DataMerger()
        df       = merger._step2_vaccine_completeness(iz_clean)
        df       = merger._step3_build_target(df)
        df["county_encoded"]     = 0
        df["sub_county_encoded"] = 0

        fp   = FeaturePipeline("config/model_config.yaml")
        X, y = fp.select_features(df)
        pre  = fp.build_preprocessor(X)
        feat = fp._all_feature_cols()

        trainer = IZDefaulterTrainer("config/model_config.yaml")
        trainer.calibration_cfg["enabled"] = False
        result  = trainer.train(X, y, pre, feat, run_name="unit_shap")

        shap_exp = SHAPExplainer("config/model_config.yaml")
        X_t      = result["preprocessor"].transform(X)
        shap_exp.fit(result["model"], X_t, feat)

        payload = shap_exp.explain_patient(
            patient_row  = X.iloc[0],
            preprocessor = result["preprocessor"],
            model        = result["model"],
            patient_meta = {"patient_id": "test_001", "patient_name": "Test Child"},
        )

        required = {"patient_id", "risk_score", "risk_tier", "top_drivers", "recommended_action"}
        assert required.issubset(payload.keys())
        assert 0.0 <= payload["risk_score"] <= 1.0
        assert payload["risk_tier"] in ("LOW", "MEDIUM", "HIGH")
        assert len(payload["top_drivers"]) > 0

    def test_risk_tier_logic(self):
        from src.explainability.shap_explainer import SHAPExplainer
        exp = SHAPExplainer("config/model_config.yaml")
        assert exp._risk_tier(0.10) == "LOW"
        assert exp._risk_tier(0.45) == "MEDIUM"
        assert exp._risk_tier(0.80) == "HIGH"


# ══ Drift Detection Tests ═════════════════════════════════════════════════════

class TestDriftDetector:
    def test_psi_zero_for_identical_distributions(self):
        from src.monitoring.drift_detector import DriftDetector
        X = pd.DataFrame({"a": np.random.normal(0, 1, 100)})
        y = pd.Series(np.random.randint(0, 2, 100))
        det = DriftDetector()
        det.fit_reference(X, y)
        df  = det.detect(X, y)
        # PSI on identical data should be small
        assert df["psi"].max() < 0.5

    def test_psi_alert_for_shifted_distribution(self):
        from src.monitoring.drift_detector import DriftDetector
        X_ref = pd.DataFrame({"a": np.random.normal(0, 1, 200)})
        X_new = pd.DataFrame({"a": np.random.normal(5, 1, 200)})  # large shift
        y_ref = pd.Series(np.zeros(200))
        det   = DriftDetector()
        det.fit_reference(X_ref, y_ref)
        df    = det.detect(X_new)
        assert df["psi"].max() > 0.1

    def test_report_html_non_empty(self):
        from src.monitoring.drift_detector import DriftDetector
        X = pd.DataFrame({"a": [1, 2, 3]})
        y = pd.Series([0, 1, 0])
        det = DriftDetector()
        det.fit_reference(X, y)
        df  = det.detect(X, y)
        html = det.report_html(df)
        assert "<table" in html
        assert "PSI" in html
