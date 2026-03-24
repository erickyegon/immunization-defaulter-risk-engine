from __future__ import annotations

from pathlib import Path
import pandas as pd
from src.validation.drift_checks import simple_missingness_profile


def build_data_quality_report(df: pd.DataFrame, output_path: str | Path) -> pd.DataFrame:
    report = simple_missingness_profile(df)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(output_path, index=False)
    return report
