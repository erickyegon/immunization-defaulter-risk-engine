"""
etl/cleaner.py
─────────────────────────────────────────────────────────────────────────────
Table-specific cleaning logic addressing every issue identified in the data
audit: sentinel values, percentage infinities, mixed boolean types, date
standardisation, JSON column parsing, and null recoding.
"""

import json
import logging
import re
from typing import Dict

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


class DataCleaner:
    """
    Applies domain-aware cleaning to each table independently.
    All operations are deterministic and logged.
    """

    def clean_all(self, tables: Dict[str, pd.DataFrame]) -> Dict[str, pd.DataFrame]:
        cleaned = {}
        dispatch = {
            "iz":           self.clean_iz,
            "active_chps":  self.clean_active_chps,
            "supervision":  self.clean_supervision,
            "homevisit":    self.clean_homevisit,
            "population":   self.clean_population,
            "pnc":          self.clean_pnc,
            "preg_reg":     self.clean_preg_reg,
            "preg_reg2":    self.clean_preg_reg,   # same schema
            "preg_visit":   self.clean_preg_visit,
            "preg_visit2":  self.clean_preg_visit, # same schema
            "fp":           self.clean_fp,
            "refill":       self.clean_refill,
        }
        for name, df in tables.items():
            fn = dispatch.get(name, lambda x: x)
            cleaned[name] = fn(df.copy())
            logger.info(f"  Cleaned {name:15s} → {cleaned[name].shape}")
        return cleaned

    # ── IZ (core child immunization) ───────────────────────────────────────

    def clean_iz(self, df: pd.DataFrame) -> pd.DataFrame:
        # Standardise month column (Feb25_2 → 2025-02, Dec24 → 2024-12)
        df["month_clean"] = df["month"].apply(self._parse_month)

        # Coerce core numeric columns (PostgreSQL may deliver these as strings)
        for col in ["patient_age_in_months", "patient_age_in_years", "patient_age_in_days"]:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")

        # Sentinel fix: due_count = -1 means "child too young — not applicable"
        df["due_count_clean"] = pd.to_numeric(df["due_count"], errors="coerce")
        df["due_count_clean"] = df["due_count_clean"].where(
            df["due_count_clean"] >= 0, other=np.nan
        )

        # Binary encode yes/no columns
        yn_cols = [
            "needs_follow_up", "needs_immunization_follow_up",
            "has_good_immunization_status", "is_available",
            "is_in_malaria_endemic_region", "is_participating_in_monthly_growth_monitoring",
            "has_signs_of_delayed_milestones", "is_participating_in_growth_monitoring",
        ]
        for col in yn_cols:
            if col in df.columns:
                df[f"{col}_binary"] = (
                    df[col].str.lower().str.strip().map({"yes": 1, "no": 0})
                )

        # Recode patient sex
        df["patient_sex_binary"] = df["patient_sex"].str.lower().map(
            {"male": 1, "female": 0, "m": 1, "f": 0}
        )

        # Malaria vaccine nulls: if not endemic → recode to 0
        malaria_vax = [
            "has_malaria_6_vaccine", "has_malaria_7_vaccine",
            "has_malaria_8_vaccine", "has_malaria_24_vaccine",
        ]
        endemic_flag = df.get("is_in_malaria_endemic_region", pd.Series("no", index=df.index))
        not_endemic = endemic_flag.str.lower().isin(["no"]) | endemic_flag.isna()
        for col in malaria_vax:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce")
                df.loc[not_endemic & df[col].isna(), col] = 0

        # Parse JSON screening column
        if "screening" in df.columns:
            df["screening_parsed"] = df["screening"].apply(self._parse_json_safe)

        # Drop entirely-null administrative columns not used in modeling
        drop_if_null = ["vaccination_upto_date"]
        for col in drop_if_null:
            if col in df.columns and df[col].isna().all():
                df.drop(columns=[col], inplace=True)
                logger.debug(f"  iz: dropped 100%-null column '{col}'")

        # Parse reported date
        df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)

        return df

    # ── Active CHPs (CHW roster) ───────────────────────────────────────────

    def clean_active_chps(self, df: pd.DataFrame) -> pd.DataFrame:
        df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)
        # Ensure phone is string for display (not numeric)
        df["chw_phone"] = df["chw_phone"].astype(str)
        return df

    # ── Supervision ────────────────────────────────────────────────────────

    def clean_supervision(self, df: pd.DataFrame) -> pd.DataFrame:
        df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)

        # Critical: fix perce_campaign_service_score = Infinity (division by zero)
        perce_cols = [c for c in df.columns if c.startswith("perce_")]
        for col in perce_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
            df[col] = df[col].replace([np.inf, -np.inf], np.nan)
            df[col] = df[col].clip(0, 1)

        # Drop inactive programme denominators (all 0 — not relevant modules)
        inactive = [
            "calc_cancer_denominator", "calc_oral_health_denominator",
            "calc_eye_health_denominator", "calc_cancer_score",
            "calc_oral_health_score", "calc_eye_health_score",
        ]
        df.drop(columns=[c for c in inactive if c in df.columns], inplace=True)

        # Safe score percentages (guarded division)
        score_pairs = [
            ("calc_immunization_score",            "calc_immunization_denominator"),
            ("calc_assessment_score",              "calc_assessment_denominator"),
            ("calc_family_planning_score",         "calc_family_planning_denominator"),
            ("calc_nutrition_score",               "calc_nutrition_denominator"),
            ("calc_pregnancy_home_visit_score",    "calc_pregnancy_home_visit_denominator"),
            ("calc_newborn_visit_score",           "calc_newborn_visit_denominator"),
            ("calc_wash_score",                    "calc_wash_denominator"),
        ]
        for num, den in score_pairs:
            if num in df.columns and den in df.columns:
                pct_col = num.replace("calc_", "pct_").replace("_score", "_pct")
                num_vals = pd.to_numeric(df[num], errors="coerce")
                den_vals = pd.to_numeric(df[den], errors="coerce")
                df[pct_col] = np.where(
                    den_vals > 0,
                    num_vals / den_vals,
                    np.nan
                ).clip(0, 1)

        # Boolean tools columns
        for col in ["has_all_tools", "has_proper_protective_equipment", "has_essential_medicines"]:
            if col in df.columns:
                df[f"{col}_binary"] = df[col].map(
                    {"yes": 1, "no": 0, True: 1, False: 0}
                ).where(df[col].notna())

        return df

    # ── Home visit ─────────────────────────────────────────────────────────

    def clean_homevisit(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pre-aggregated (postgres): only chw_area + rate columns — nothing to clean
        if "reported" not in df.columns:
            return df
        df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)
        df["year_month"] = df["reported"].dt.to_period("M").astype(str)
        return df

    # ── Population ─────────────────────────────────────────────────────────

    def clean_population(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pre-aggregated (postgres): only (chw_area, chw_workload_u2) — nothing to clean
        if "chw_workload_u2" in df.columns:
            return df
        # Raw CSV: drop unused columns and coerce u2_pop
        cols_keep = ["county", "month", "chw_area", "chw_uuid", "u2_pop", "reportedm"]
        existing = [c for c in cols_keep if c in df.columns]
        df = df[existing].copy()
        df["u2_pop"] = pd.to_numeric(df["u2_pop"], errors="coerce")
        return df

    # ── PNC (postnatal care) ────────────────────────────────────────────────

    def clean_pnc(self, df: pd.DataFrame) -> pd.DataFrame:
        df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)

        # Sentinel: days_since_delivery = -1 means pre-delivery record
        df["days_since_delivery_clean"] = pd.to_numeric(
            df.get("days_since_delivery"), errors="coerce"
        )
        df["days_since_delivery_clean"] = df["days_since_delivery_clean"].where(
            df["days_since_delivery_clean"] >= 0, other=np.nan
        )

        # Target validation label
        df["is_immunization_defaulter_binary"] = df.get(
            "is_immunization_defaulter", pd.Series(dtype=str)
        ).str.lower().map({"yes": 1, "no": 0})

        return df

    # ── Pregnancy registration (shared logic for preg_reg + preg_reg2) ─────

    def clean_preg_reg(self, df: pd.DataFrame) -> pd.DataFrame:
        df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)
        df["date_of_birth"] = pd.to_datetime(df["date_of_birth"], errors="coerce")

        # Standardise boolean-like columns
        bool_map = {"true": 1, "false": 0, "yes": 1, "no": 0,
                    True: 1, False: 0, "1": 1, "0": 0}

        for col in ["marked_as_pregnant", "is_anc_defaulter", "currently_pregnant"]:
            if col in df.columns:
                df[f"{col}_binary"] = df[col].astype(str).str.lower().map(bool_map)

        # Muac risk: anything not 'green' is nutritional risk
        if "muac_color" in df.columns:
            df["muac_risk"] = (~df["muac_color"].str.lower().eq("green")).astype(float)
            df.loc[df["muac_color"].isna(), "muac_risk"] = np.nan

        # Iron/folate supplement binary
        if "takes_iron_or_folate_supplements" in df.columns:
            df["iron_folate_binary"] = df["takes_iron_or_folate_supplements"].str.lower().map(
                {"yes": 1, "no": 0}
            )

        # ANC attendance as numeric
        if "number_of_anc_attended" in df.columns:
            df["number_of_anc_attended"] = pd.to_numeric(
                df["number_of_anc_attended"], errors="coerce"
            )

        return df

    # ── Pregnancy visit ────────────────────────────────────────────────────

    def clean_preg_visit(self, df: pd.DataFrame) -> pd.DataFrame:
        df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)
        if "number_of_anc_attended" in df.columns:
            df["number_of_anc_attended"] = pd.to_numeric(
                df["number_of_anc_attended"], errors="coerce"
            )
        bool_map = {"true": 1, "false": 0, "yes": 1, "no": 0, True: 1, False: 0}
        for col in ["is_anc_defaulter", "currently_pregnant", "marked_as_pregnant"]:
            if col in df.columns:
                df[f"{col}_binary"] = df[col].astype(str).str.lower().map(bool_map)
        return df

    # ── Family planning ────────────────────────────────────────────────────

    def clean_fp(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pre-aggregated (postgres): only (contact_parent_id, on_fp_binary) — nothing to clean
        if "on_fp_binary" in df.columns and "reported" not in df.columns:
            return df
        if "reported" in df.columns:
            df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)
        if "on_fp" in df.columns:
            df["on_fp_binary"] = df["on_fp"].str.lower().map({"yes": 1, "no": 0})
        return df

    # ── Refill ─────────────────────────────────────────────────────────────

    def clean_refill(self, df: pd.DataFrame) -> pd.DataFrame:
        # Pre-aggregated (postgres): only (contact_parent_id, on_fp_binary) — nothing to clean
        if "on_fp_binary" in df.columns and "reported" not in df.columns:
            return df
        if "reported" in df.columns:
            df["reported"] = pd.to_datetime(df["reported"], errors="coerce", utc=True)
        if "on_fp" in df.columns:
            df["on_fp_binary"] = df["on_fp"].str.lower().map({"yes": 1, "no": 0})
        return df

    # ── Helpers ────────────────────────────────────────────────────────────

    @staticmethod
    def _parse_month(s: str) -> str:
        """Convert 'Feb25_2', 'Feb25', 'Dec24' → 'YYYY-MM' string."""
        if pd.isna(s):
            return np.nan
        s = str(s).strip()
        # Pattern: 3-letter month + 2-digit year (optional suffix)
        m = re.match(r"([A-Za-z]{3})(\d{2})", s)
        if m:
            month_str, year_str = m.group(1), m.group(2)
            month_map = {
                "jan": "01", "feb": "02", "mar": "03", "apr": "04",
                "may": "05", "jun": "06", "jul": "07", "aug": "08",
                "sep": "09", "oct": "10", "nov": "11", "dec": "12",
            }
            mm = month_map.get(month_str.lower(), "00")
            yy = f"20{year_str}"
            return f"{yy}-{mm}"
        return s

    @staticmethod
    def _parse_json_safe(val) -> dict:
        """Safely parse JSON string; return {} on failure."""
        if pd.isna(val):
            return {}
        try:
            if isinstance(val, dict):
                return val
            val = str(val).strip()
            if val.startswith("{"):
                return json.loads(val)
        except Exception:
            pass
        return {}
