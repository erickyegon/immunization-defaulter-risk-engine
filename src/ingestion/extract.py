from __future__ import annotations

import time
from pathlib import Path
from typing import Iterable

import pandas as pd
from sqlalchemy import text

from src.ingestion.db import get_engine
from src.utils.logging import get_logger

logger = get_logger(__name__)

RAW_DIR = Path("data/raw")
DEFAULT_TABLES = [
    "iz",
    "supervision",
    "active_chps",
    "homevisit",
    "population",
]


def _validate_table_name(table_name: str) -> str:
    cleaned = table_name.strip()
    if cleaned not in DEFAULT_TABLES:
        raise ValueError(
            f"Unsupported table '{table_name}'. Allowed tables: {DEFAULT_TABLES}"
        )
    return cleaned


def get_table_row_count(table_name: str) -> int:
    """
    Return total row count for a source table.
    Useful for diagnostics before extraction.
    """
    validated_table = _validate_table_name(table_name)
    engine = get_engine()
    query = text(f"SELECT COUNT(*) AS row_count FROM public.{validated_table}")

    start = time.time()
    with engine.connect() as conn:
        row_count = int(conn.execute(query).scalar() or 0)

    elapsed = time.time() - start
    logger.info(
        "Row count for table %s = %s (%.2f sec)",
        validated_table,
        row_count,
        elapsed,
    )
    return row_count


def extract_table(
    table_name: str,
    chunksize: int = 5000,
    output_format: str = "csv",
) -> pd.DataFrame:
    """
    Extract a PostgreSQL table in chunks and write it to disk progressively.

    Parameters
    ----------
    table_name : str
        Source PostgreSQL table name in public schema.
    chunksize : int
        Number of rows to stream per chunk.
    output_format : str
        Currently supports 'csv'. Parquet can be added later.

    Returns
    -------
    pd.DataFrame
        Concatenated dataframe for downstream immediate use.
        For very large tables, you may prefer not to rely on the returned dataframe.
    """
    validated_table = _validate_table_name(table_name)
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    if output_format.lower() != "csv":
        raise ValueError("Only 'csv' output_format is currently supported.")

    output_path = RAW_DIR / f"{validated_table}.csv"
    query = f"SELECT * FROM public.{validated_table}"
    engine = get_engine()

    logger.info(
        "Starting extraction for table %s with chunksize=%s",
        validated_table,
        chunksize,
    )

    total_start = time.time()
    total_rows = 0
    first_chunk = True
    chunk_frames: list[pd.DataFrame] = []

    # Remove existing output so repeated runs do not append stale data.
    if output_path.exists():
        output_path.unlink()

    try:
        for chunk_number, chunk_df in enumerate(
            pd.read_sql(query, engine, chunksize=chunksize),
            start=1,
        ):
            chunk_start = time.time()
            rows_in_chunk = len(chunk_df)
            total_rows += rows_in_chunk

            chunk_df.to_csv(
                output_path,
                mode="w" if first_chunk else "a",
                header=first_chunk,
                index=False,
            )
            first_chunk = False

            chunk_frames.append(chunk_df)

            logger.info(
                (
                    "Extracted table %s | chunk=%s | rows_in_chunk=%s | "
                    "cumulative_rows=%s | chunk_time=%.2f sec"
                ),
                validated_table,
                chunk_number,
                rows_in_chunk,
                total_rows,
                time.time() - chunk_start,
            )

        total_elapsed = time.time() - total_start
        logger.info(
            "Finished extraction for table %s | total_rows=%s | saved_to=%s | total_time=%.2f sec",
            validated_table,
            total_rows,
            output_path.as_posix(),
            total_elapsed,
        )

        if chunk_frames:
            return pd.concat(chunk_frames, ignore_index=True)

        logger.warning("Table %s returned zero rows", validated_table)
        return pd.DataFrame()

    except Exception:
        logger.exception("Extraction failed for table %s", validated_table)
        raise


def extract_selected_sources(
    table_names: Iterable[str],
    chunksize: int = 5000,
) -> dict[str, pd.DataFrame]:
    """
    Extract only selected source tables.
    """
    extracted: dict[str, pd.DataFrame] = {}

    for table_name in table_names:
        validated_table = _validate_table_name(table_name)
        logger.info("Preparing selected extraction for table %s", validated_table)

        row_count = get_table_row_count(validated_table)
        logger.info("Proceeding to extract table %s with %s rows", validated_table, row_count)

        extracted[validated_table] = extract_table(
            table_name=validated_table,
            chunksize=chunksize,
        )

    return extracted


def extract_all_sources(chunksize: int = 5000) -> dict[str, pd.DataFrame]:
    """
    Extract all configured source tables with row-count diagnostics.
    """
    extracted: dict[str, pd.DataFrame] = {}

    for table_name in DEFAULT_TABLES:
        logger.info("Preparing extraction for table %s", table_name)

        row_count = get_table_row_count(table_name)
        logger.info("Proceeding to extract table %s with %s rows", table_name, row_count)

        extracted[table_name] = extract_table(
            table_name=table_name,
            chunksize=chunksize,
        )

    return extracted