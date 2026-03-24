from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.utils.io import write_table
from src.utils.logging import get_logger

logger = get_logger(__name__)


def snapshot_dataframe(df: pd.DataFrame, output_path: str | Path) -> None:
    write_table(df, output_path)
    logger.info("Saved snapshot to %s with %s rows", output_path, len(df))
