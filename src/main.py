from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from src.cleaning.chw_cleaning import clean_active_chps
from src.cleaning.homevisit_cleaning import clean_homevisit
from src.cleaning.iz_cleaning import clean_iz
from src.cleaning.population_cleaning import clean_population
from src.cleaning.supervision_cleaning import clean_supervision
from src.explainability.shap_global import generate_global_shap_report
from src.explainability.shap_local import generate_local_shap_report
from src.features.feature_store import build_feature_store
from src.ingestion.extract import extract_all_sources, extract_selected_sources
from src.labeling.cohort_builder import build_child_month_cohort
from src.labeling.target_builder import build_labels
from src.modeling.evaluate import evaluate_saved_model
from src.modeling.predict import score_latest_cohort
from src.modeling.train import train_model
from src.serving.export_outputs import export_serving_outputs
from src.serving.risk_lists import build_risk_lists
from src.utils.logging import get_logger
from src.validation.data_quality_report import generate_data_quality_report

logger = get_logger(__name__)


def ensure_directories() -> None:
    required_dirs = [
        Path("data/raw"),
        Path("data/interim"),
        Path("data/processed"),
        Path("data/outputs"),
        Path("models"),
        Path("reports/data_quality"),
        Path("reports/model_performance"),
        Path("reports/shap"),
        Path("reports/deployment_validation"),
    ]
    for directory in required_dirs:
        directory.mkdir(parents=True, exist_ok=True)


def run_extract(table_names: list[str] | None = None, chunksize: int = 5000) -> None:
    start = time.time()
    logger.info("Starting extraction stage")
    if table_names:
        extract_selected_sources(table_names=table_names, chunksize=chunksize)
    else:
        extract_all_sources(chunksize=chunksize)
    logger.info("Finished extraction stage in %.2f seconds", time.time() - start)


def run_clean() -> None:
    start = time.time()
    logger.info("Starting cleaning stage")

    clean_iz()
    clean_supervision()
    clean_active_chps()
    clean_homevisit()
    clean_population()

    logger.info("Finished cleaning stage in %.2f seconds", time.time() - start)


def run_label() -> None:
    start = time.time()
    logger.info("Starting cohort and labeling stage")

    build_child_month_cohort()
    build_labels()

    logger.info("Finished cohort and labeling stage in %.2f seconds", time.time() - start)


def run_features() -> None:
    start = time.time()
    logger.info("Starting feature engineering stage")

    build_feature_store()

    logger.info("Finished feature engineering stage in %.2f seconds", time.time() - start)


def run_train() -> None:
    start = time.time()
    logger.info("Starting model training stage")

    train_model()

    logger.info("Finished model training stage in %.2f seconds", time.time() - start)


def run_evaluate() -> None:
    start = time.time()
    logger.info("Starting evaluation stage")

    evaluate_saved_model()

    logger.info("Finished evaluation stage in %.2f seconds", time.time() - start)


def run_explain() -> None:
    start = time.time()
    logger.info("Starting explainability stage")

    generate_global_shap_report()
    generate_local_shap_report()

    logger.info("Finished explainability stage in %.2f seconds", time.time() - start)


def run_serve() -> None:
    start = time.time()
    logger.info("Starting serving/export stage")

    score_latest_cohort()
    build_risk_lists()
    export_serving_outputs()

    logger.info("Finished serving/export stage in %.2f seconds", time.time() - start)


def run_report() -> None:
    start = time.time()
    logger.info("Starting data quality reporting stage")

    generate_data_quality_report()

    logger.info("Finished data quality reporting stage in %.2f seconds", time.time() - start)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Immunization Defaulter Risk Engine pipeline runner"
    )
    parser.add_argument(
        "--stage",
        required=True,
        choices=[
            "extract",
            "clean",
            "label",
            "features",
            "train",
            "evaluate",
            "explain",
            "serve",
            "report",
            "full",
            "full_with_extract",
        ],
        help=(
            "Pipeline stage to run. "
            "'full' runs clean -> label -> features -> train -> evaluate -> explain -> serve -> report. "
            "'full_with_extract' runs extract first, then the rest."
        ),
    )
    parser.add_argument(
        "--tables",
        nargs="*",
        default=None,
        help=(
            "Optional list of source tables to extract, e.g. "
            "--tables iz supervision active_chps"
        ),
    )
    parser.add_argument(
        "--chunksize",
        type=int,
        default=5000,
        help="Chunk size for SQL extraction. Default: 5000",
    )
    return parser.parse_args()


def main() -> int:
    ensure_directories()
    args = parse_args()

    try:
        if args.stage == "extract":
            run_extract(table_names=args.tables, chunksize=args.chunksize)

        elif args.stage == "clean":
            run_clean()

        elif args.stage == "label":
            run_label()

        elif args.stage == "features":
            run_features()

        elif args.stage == "train":
            run_train()

        elif args.stage == "evaluate":
            run_evaluate()

        elif args.stage == "explain":
            run_explain()

        elif args.stage == "serve":
            run_serve()

        elif args.stage == "report":
            run_report()

        elif args.stage == "full":
            run_clean()
            run_label()
            run_features()
            run_train()
            run_evaluate()
            run_explain()
            run_serve()
            run_report()

        elif args.stage == "full_with_extract":
            run_extract(table_names=args.tables, chunksize=args.chunksize)
            run_clean()
            run_label()
            run_features()
            run_train()
            run_evaluate()
            run_explain()
            run_serve()
            run_report()

        else:
            logger.error("Unsupported stage: %s", args.stage)
            return 1

        logger.info("Pipeline completed successfully")
        return 0

    except KeyboardInterrupt:
        logger.warning("Pipeline interrupted by user")
        return 130

    except Exception as exc:
        logger.exception("Pipeline failed: %s", exc)
        return 1


if __name__ == "__main__":
    sys.exit(main())