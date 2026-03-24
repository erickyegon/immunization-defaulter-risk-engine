"""
etl/loader.py
─────────────────────────────────────────────────────────────────────────────
Unified data loader supporting both CSV extracts (dev/portfolio) and live
PostgreSQL connections (production). Swap source via config/model_config.yaml.
"""

import os
import logging
from pathlib import Path
from typing import Dict, Optional

import pandas as pd
import yaml

logger = logging.getLogger(__name__)

# Canonical table → filename mapping
TABLE_MAP = {
    "iz":           "iz.csv",
    "active_chps":  "active_chps.csv",
    "supervision":  "supervision.csv",
    "homevisit":    "homevisit.csv",
    "population":   "population.csv",
    "pnc":          "pnc.csv",
    "preg_reg":     "preg_reg.csv",
    "preg_reg2":    "preg_reg2.csv",
    "preg_visit":   "preg_visit.csv",
    "preg_visit2":  "preg_visit2.csv",
    "fp":           "fp.csv",
    "refill":       "refill.csv",
}

# Targeted column lists for large tables — avoids SELECT * on multi-GB tables.
# None means SELECT * (table is small enough to load fully).
POSTGRES_COLUMNS = {
    # 3.8 GB — merger only uses 4 identifier/geo columns + cleaner uses reported+phone
    "active_chps": [
        "chw_area_uuid", "community_unit", "county_name",
        "sub_county_name", "reported", "chw_phone",
    ],
    # 805 MB — merger only groups by chw_area+reported and counts family_id
    "homevisit": ["chw_area", "reported", "family_id"],
    # 659 MB — main.py aggregates on_fp_binary per contact_parent_id
    "fp": ["contact_parent_id", "on_fp", "reported"],
    # 584 MB — cleaner immediately drops all but these 6 columns
    "population": ["county", "month", "chw_area", "chw_uuid", "u2_pop", "reportedm"],
}

# Columns to parse as dates across tables
DATE_COLUMNS = {
    "iz":           ["reported", "immunization_follow_up_date", "follow_up_date"],
    "supervision":  ["reported", "next_supervision_visit_date"],
    "homevisit":    ["reported"],
    "preg_reg":     ["reported", "last_anc_date", "next_anc_visit_date", "date_of_birth"],
    "preg_reg2":    ["reported", "last_anc_date", "next_anc_visit_date", "date_of_birth"],
    "preg_visit":   ["reported", "next_anc_visit_date"],
    "preg_visit2":  ["reported", "next_anc_visit_date"],
    "pnc":          ["reported", "date_of_delivery"],
    "fp":           ["reported"],
    "refill":       ["reported"],
}


class DataLoader:
    """
    Loads all 12 CHW platform tables into a dictionary of DataFrames.

    Parameters
    ----------
    config_path : str | Path
        Path to model_config.yaml
    raw_dir : str | Path | None
        Override raw data directory (defaults to config value)
    """

    def __init__(self, config_path: str = "config/model_config.yaml",
                 raw_dir: Optional[str] = None):
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        self.source = self.cfg["data"]["source"]
        self.raw_dir = Path(raw_dir or self.cfg["data"]["raw_dir"])
        logger.info(f"DataLoader initialised | source={self.source} | dir={self.raw_dir}")

    def load_all(self) -> Dict[str, pd.DataFrame]:
        """Return dict of {table_name: DataFrame} for all 12 tables."""
        if self.source == "csv":
            return self._load_csv_all()
        elif self.source == "postgres":
            return self._load_postgres_all()
        else:
            raise ValueError(f"Unknown source: {self.source}")

    def load_table(self, table: str) -> pd.DataFrame:
        """Load a single table by canonical name."""
        if self.source == "csv":
            return self._load_csv(table)
        return self._load_postgres(table)

    # ── CSV backend ────────────────────────────────────────────────────────

    def _load_csv_all(self) -> Dict[str, pd.DataFrame]:
        tables = {}
        for name in TABLE_MAP:
            try:
                tables[name] = self._load_csv(name)
                logger.info(f"  Loaded {name:15s} → {tables[name].shape}")
            except FileNotFoundError:
                logger.warning(f"  Table {name} not found — skipping")
        return tables

    def _load_csv(self, table: str) -> pd.DataFrame:
        path = self.raw_dir / TABLE_MAP[table]
        if not path.exists():
            raise FileNotFoundError(f"CSV not found: {path}")

        parse_dates = DATE_COLUMNS.get(table, [])
        df = pd.read_csv(
            path,
            parse_dates=[c for c in parse_dates],  # only parse existing cols
            low_memory=False,
        )
        df = self._strip_whitespace(df)
        df["_source_table"] = table
        return df

    # ── PostgreSQL backend ─────────────────────────────────────────────────

    def _get_engine(self):
        """Return a SQLAlchemy engine built from POSTGRES_* env vars."""
        from src.ingestion.db import get_engine
        return get_engine()

    def _load_postgres_all(self) -> Dict[str, pd.DataFrame]:
        """Load all tables from PostgreSQL using POSTGRES_* env vars."""
        engine = self._get_engine()
        schema = os.getenv("POSTGRES_SCHEMA", "public")
        tables = {}
        for name in TABLE_MAP:
            sql = self._build_query(name, schema)
            logger.info(f"  Querying {name} ...")
            try:
                df = pd.read_sql(sql, engine)
                df = self._strip_whitespace(df)
                df = self._parse_dates(df, name)
                df = df.assign(_source_table=name)
                tables[name] = df
                logger.info(f"  Loaded  {name:15s} from postgres → {df.shape}")
            except Exception as e:
                logger.error(f"  Failed to load {name}: {e}")
        if not tables:
            raise RuntimeError(
                "No tables loaded from PostgreSQL. "
                "Check POSTGRES_* env vars and that the database is reachable."
            )
        return tables

    def _load_postgres(self, table: str) -> pd.DataFrame:
        engine = self._get_engine()
        schema = os.getenv("POSTGRES_SCHEMA", "public")
        df = pd.read_sql(self._build_query(table, schema), engine)
        df = self._strip_whitespace(df)
        df = self._parse_dates(df, table)
        return df

    def _build_query(self, table: str, schema: str = "public") -> str:
        """
        Returns the most efficient SQL for each table:
        - Small tables              → SELECT *
        - Dedup tables              → DISTINCT ON (patient_id, reportedm)
        - Large lookup tables       → DISTINCT ON (key column) — one row per entity
        - Large time-series tables  → GROUP BY aggregation pushed to DB
        Results for pre-aggregated tables already have the columns the merger expects,
        so Python groupby steps are skipped automatically.
        """
        q = f'"{schema}"."{table}"'

        # ── Dedup tables (keep latest record per patient per month) ───────────
        if table in {"iz", "preg_reg", "preg_reg2", "preg_visit", "preg_visit2"}:
            return f"""
                SELECT DISTINCT ON (patient_id, reportedm) *
                FROM {q}
                ORDER BY patient_id, reportedm, reported DESC
            """

        # ── active_chps: 11 M rows → one row per CHW area ─────────────────────
        if table == "active_chps":
            return f"""
                SELECT DISTINCT ON (chw_area_uuid)
                    chw_area_uuid, community_unit, county_name,
                    sub_county_name, reported, chw_phone
                FROM {q}
                ORDER BY chw_area_uuid, reported DESC
            """

        # ── homevisit: 3.3 M rows → monthly visit rate per CHW area ──────────
        if table == "homevisit":
            return f"""
                SELECT
                    chw_area,
                    COUNT(family_id)                                            AS total_homevisits,
                    COUNT(DISTINCT DATE_TRUNC('month', reported::timestamp))    AS hv_distinct_months,
                    COUNT(family_id)::float /
                        NULLIF(COUNT(DISTINCT DATE_TRUNC('month', reported::timestamp)), 0)
                                                                                AS monthly_homevisit_rate
                FROM {q}
                GROUP BY chw_area
            """

        # ── population: 2.6 M rows → median u2 workload per CHW area ─────────
        if table == "population":
            return f"""
                SELECT
                    chw_area,
                    PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY u2_pop) AS chw_workload_u2
                FROM {q}
                WHERE u2_pop IS NOT NULL
                GROUP BY chw_area
            """

        # ── fp / refill: 1 M rows → max on_fp per household ──────────────────
        if table in {"fp", "refill"}:
            return f"""
                SELECT
                    contact_parent_id,
                    MAX(CASE WHEN LOWER(TRIM(on_fp)) = 'yes' THEN 1 ELSE 0 END) AS on_fp_binary
                FROM {q}
                WHERE contact_parent_id IS NOT NULL
                GROUP BY contact_parent_id
            """

        # ── All other tables: full load (they are small) ───────────────────────
        return f"SELECT * FROM {q}"

    def _parse_dates(self, df: pd.DataFrame, table: str) -> pd.DataFrame:
        """Parse known date columns to datetime after loading from postgres."""
        for col in DATE_COLUMNS.get(table, []):
            if col in df.columns:
                df[col] = pd.to_datetime(df[col], errors="coerce")
        return df

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _strip_whitespace(df: pd.DataFrame) -> pd.DataFrame:
        """Strip leading/trailing whitespace from string columns (safe)."""
        for col in df.select_dtypes(include="object").columns:
            try:
                df[col] = df[col].str.strip()
            except (AttributeError, TypeError):
                pass
        return df

    def summary(self, tables: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Print a load summary DataFrame."""
        rows = []
        for name, df in tables.items():
            rows.append({
                "table": name,
                "rows": len(df),
                "cols": len(df.columns),
                "null_pct": round(df.isnull().mean().mean() * 100, 1),
            })
        return pd.DataFrame(rows)
