"""
main.py
─────────────────────────────────────────────────────────────────────────────
CLI entry point for the IZ Defaulter Prediction pipeline.

Usage:
  python main.py --stage all          # full pipeline
  python main.py --stage etl          # ETL only
  python main.py --stage train        # train (requires processed data)
  python main.py --stage evaluate     # evaluate + SHAP
  python main.py --stage monitor      # drift detection demo
  python main.py --stage api          # launch FastAPI server
"""

import argparse
import logging
import os
import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split as _train_test_split

# ── Path setup so imports work from project root ───────────────────────────
PROJECT_ROOT = Path(__file__).parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s | %(levelname)-8s | %(message)s",
    datefmt = "%H:%M:%S",
)
logger = logging.getLogger(__name__)

CONFIG_PATH = str(PROJECT_ROOT / "config" / "model_config.yaml")


# ══ Pipeline stages ═══════════════════════════════════════════════════════════

def stage_etl() -> pd.DataFrame:
    """Load, clean, merge → analytical dataset."""
    logger.info("\n" + "═"*60)
    logger.info("STAGE 1 — ETL: Load · Clean · Merge")
    logger.info("═"*60)

    from src.etl.loader  import DataLoader
    from src.etl.cleaner import DataCleaner
    from src.etl.merger  import DataMerger

    # Load
    loader = DataLoader(CONFIG_PATH, raw_dir=str(PROJECT_ROOT / "data" / "raw"))
    tables = loader.load_all()
    logger.info(f"\nLoad summary:\n{loader.summary(tables).to_string(index=False)}")

    # Clean
    cleaner = DataCleaner()
    tables  = cleaner.clean_all(tables)

    # Add vitamin A completeness to iz before merge
    from src.features.pipeline import compute_vitamin_a_completeness
    if "iz" in tables:
        tables["iz"]["vitamin_a_completeness"] = compute_vitamin_a_completeness(tables["iz"])
        # Add household FP flag from fp/refill
        fp_on = []
        for tbl in ["fp", "refill"]:
            if tbl not in tables:
                continue
            tbl_df = tables[tbl]
            # Postgres path: SQL already returned (contact_parent_id, on_fp_binary)
            # CSV path: cleaner has already mapped on_fp → on_fp_binary
            if "on_fp_binary" in tbl_df.columns and "contact_parent_id" in tbl_df.columns:
                agg = (
                    tbl_df
                    .groupby("contact_parent_id", as_index=False)["on_fp_binary"]
                    .max()
                    .rename(columns={"contact_parent_id": "chw_area_key",
                                     "on_fp_binary": "household_on_fp"})
                )
                fp_on.append(agg)
        if fp_on:
            fp_df = pd.concat(fp_on).groupby("chw_area_key")["household_on_fp"].max().reset_index()
            tables["iz"] = tables["iz"].merge(fp_df, left_on="contact_parent_id",
                                              right_on="chw_area_key", how="left")
        # Recode is_in_malaria_endemic_region to binary
        if "is_in_malaria_endemic_region" in tables["iz"].columns:
            tables["iz"]["is_malaria_endemic_binary"] = (
                tables["iz"]["is_in_malaria_endemic_region"].str.lower()
                .map({"yes": 1, "no": 0})
            )

    # Merge
    merger  = DataMerger()
    df      = merger.build_analytical_dataset(tables)

    # Save
    out_dir = PROJECT_ROOT / "data" / "processed"
    out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_dir / "analytical_dataset.parquet", index=False)
    logger.info(f"\nSaved analytical dataset → {out_dir}/analytical_dataset.parquet")
    return df


def stage_train(df: pd.DataFrame = None):
    """Feature engineering → XGBoost training → MLflow logging."""
    logger.info("\n" + "═"*60)
    logger.info("STAGE 2 — TRAIN: Features · XGBoost · Calibration · MLflow")
    logger.info("═"*60)

    from src.features.pipeline import FeaturePipeline
    from src.model.trainer     import IZDefaulterTrainer
    from src.model.tuner       import HyperparameterTuner

    if df is None:
        path = PROJECT_ROOT / "data" / "processed" / "analytical_dataset.parquet"
        df   = pd.read_parquet(path)
        logger.info(f"Loaded analytical dataset: {df.shape}")

    # Features
    fp = FeaturePipeline(CONFIG_PATH)
    X, y = fp.select_features(df)

    logger.info(f"\nFeature report (top 10 by null %):\n"
                f"{fp.report(X, y).head(10).to_string(index=False)}")

    preprocessor  = fp.build_preprocessor(X)
    feature_names = fp._all_feature_cols()

    # Optuna tuning (only if enough data and config enabled)
    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    best_params = {}
    if cfg["tuning"]["enabled"] and len(X) >= 20:
        # Split BEFORE fitting preprocessor to avoid leaking test statistics
        # into imputer/scaler during hyperparameter search.
        X_tune_tr, _, y_tune_tr, _ = _train_test_split(
            X, y,
            test_size   = cfg["evaluation"]["test_size"],
            stratify    = y if y.nunique() > 1 else None,
            random_state= 42,
        )
        preprocessor_tune = fp.build_preprocessor(X_tune_tr)
        X_t = preprocessor_tune.fit_transform(X_tune_tr)

        n_pos = int(y_tune_tr.sum())
        n_neg = int((y_tune_tr == 0).sum())
        spw   = n_neg / max(n_pos, 1)
        tuner = HyperparameterTuner(CONFIG_PATH)
        # Limit trials for small datasets
        n_trials = min(cfg["tuning"]["n_trials"], max(5, len(X_tune_tr) // 5))
        tuner.tuning_cfg["n_trials"] = n_trials
        best_params = tuner.tune(X_t, y_tune_tr.values, scale_pos_weight=spw)
        # The tuner may have created an implicit MLflow run via log_params/log_metric;
        # end it so trainer.train() can open its own clean run.
        import mlflow as _mlflow
        _mlflow.end_run()
        # Patch config with best params
        if best_params:
            cfg["model"]["base_params"].update(best_params)

    # Train
    # Re-fit preprocessor fresh for training
    preprocessor2 = fp.build_preprocessor(X)
    trainer = IZDefaulterTrainer(CONFIG_PATH)
    result  = trainer.train(
        X, y,
        preprocessor  = preprocessor2,
        feature_names = feature_names,
        run_name      = "iz_defaulter_xgb_v1",
    )
    return result


def stage_evaluate(result: dict = None):
    """Evaluation plots + SHAP global analysis + per-patient examples."""
    logger.info("\n" + "═"*60)
    logger.info("STAGE 3 — EVALUATE: Metrics · Plots · SHAP · Fairness")
    logger.info("═"*60)

    from src.model.trainer          import IZDefaulterTrainer
    from src.model.evaluator        import ModelEvaluator
    from src.explainability.shap_explainer import SHAPExplainer
    from src.features.pipeline      import FeaturePipeline
    from sklearn.model_selection    import train_test_split

    if result is None:
        model, preprocessor, feature_names = IZDefaulterTrainer.load_artifacts(
            str(PROJECT_ROOT / "data" / "processed")
        )
        result = {"model": model, "preprocessor": preprocessor,
                  "feature_names": feature_names}

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "analytical_dataset.parquet")
    fp = FeaturePipeline(CONFIG_PATH)
    X, y = fp.select_features(df)

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=cfg["evaluation"]["test_size"],
        stratify=y if y.nunique() > 1 else None, random_state=42
    )

    model        = result["model"]
    preprocessor = result["preprocessor"]
    feat_names   = result["feature_names"]

    # Full evaluation
    evaluator = ModelEvaluator(CONFIG_PATH)
    eval_results = evaluator.full_evaluation(
        model, preprocessor, X_test, y_test, feat_names
    )

    # SHAP global
    logger.info("\n── SHAP Global Analysis ─────────────────────")
    shap_exp = SHAPExplainer(CONFIG_PATH)
    X_train_t = preprocessor.transform(X_train)
    shap_exp.fit(model, X_train_t, feat_names)

    X_test_t = preprocessor.transform(X_test)
    shap_paths = shap_exp.global_analysis(X_test_t)

    # Per-patient examples: HIGH, MEDIUM, LOW risk
    logger.info("\n── Per-Patient SHAP Examples ────────────────")
    y_prob_all = model.predict_proba(X_test_t)[:, 1]
    tiers_shown = {}

    for idx in np.argsort(y_prob_all)[::-1]:
        score = float(y_prob_all[idx])
        tier  = ("HIGH" if score >= 0.60 else
                 "MEDIUM" if score >= 0.33 else "LOW")
        if tier in tiers_shown:
            continue
        tiers_shown[tier] = True

        row_idx = X_test.index[idx]
        row     = X_test.iloc[idx]

        payload = shap_exp.explain_patient(
            patient_row = row,
            preprocessor= preprocessor,
            model       = model,
            patient_meta= {
                "patient_id":   df.loc[row_idx, "patient_id"] if "patient_id" in df.columns else str(row_idx),
                "patient_name": df.loc[row_idx, "patient_name"] if "patient_name" in df.columns else "Patient",
            },
        )

        wf_path = shap_exp.waterfall_plot(row, preprocessor, patient_id=f"{tier.lower()}_example")

        logger.info(f"\n  [{tier}] Risk={payload['risk_pct']:.1f}%")
        for d in payload["top_drivers"]:
            logger.info(f"    ↳ {d['friendly_name']}: {d['plain_english']}")
        logger.info(f"    → {payload['recommended_action']}")

        # Save JSON
        json_path = PROJECT_ROOT / "reports" / "shap" / f"patient_{tier.lower()}.json"
        with open(json_path, "w") as f:
            json.dump(payload, f, indent=2)

        if len(tiers_shown) == 3:
            break

    return eval_results


def stage_monitor():
    """Drift detection demo: compare training vs. held-out data."""
    logger.info("\n" + "═"*60)
    logger.info("STAGE 4 — MONITOR: PSI Drift Detection")
    logger.info("═"*60)

    from src.monitoring.drift_detector import DriftDetector
    from src.features.pipeline         import FeaturePipeline
    from sklearn.model_selection        import train_test_split

    df = pd.read_parquet(PROJECT_ROOT / "data" / "processed" / "analytical_dataset.parquet")
    fp = FeaturePipeline(CONFIG_PATH)
    X, y = fp.select_features(df)

    X_train, X_new, y_train, y_new = train_test_split(X, y, test_size=0.3,
                                                       stratify=y if y.nunique()>1 else None,
                                                       random_state=99)
    detector = DriftDetector(n_bins=8)
    detector.fit_reference(X_train, y_train)
    drift_df = detector.detect(X_new, y_new)

    out = PROJECT_ROOT / "reports" / "drift_report.html"
    out.write_text(detector.report_html(drift_df))
    logger.info(f"\nDrift report saved → {out}")

    # Also save CSV
    drift_df.to_csv(PROJECT_ROOT / "reports" / "drift_report.csv", index=False)
    return drift_df


def stage_api():
    """Launch FastAPI server."""
    logger.info("\n" + "═"*60)
    logger.info("STAGE 5 — API: FastAPI server")
    logger.info("═"*60)
    import uvicorn
    os.chdir(PROJECT_ROOT)
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)


# ══ CLI ═══════════════════════════════════════════════════════════════════════

def parse_args():
    parser = argparse.ArgumentParser(
        description="IZ Defaulter Prediction Pipeline"
    )
    parser.add_argument(
        "--stage",
        choices=["all", "etl", "train", "evaluate", "monitor", "api"],
        default="all",
        help="Pipeline stage to run",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    os.chdir(PROJECT_ROOT)

    if args.stage in ("all", "etl"):
        df = stage_etl()
    else:
        df = None

    if args.stage in ("all", "train"):
        result = stage_train(df)
    else:
        result = None

    if args.stage in ("all", "evaluate"):
        stage_evaluate(result)

    if args.stage in ("all", "monitor"):
        stage_monitor()

    if args.stage == "api":
        stage_api()

    logger.info("\n" + "═"*60)
    logger.info("Pipeline complete.")
    logger.info("═"*60)
