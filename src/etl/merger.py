"""
etl/merger.py
─────────────────────────────────────────────────────────────────────────────
Implements the 10-step ETL blueprint to construct the patient-level analytical
dataset from the 12-table CHW platform schema. Each step is a named method
enabling targeted testing and debugging.
"""

import logging
from typing import Dict, Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# CHW area linkage column per table (confirmed from audit)
CHW_AREA_COLS = {
    "iz":           "contact_parent_id",   # iz.contact_parent_id = chw_area_uuid
    "supervision":  "chw_area",
    "homevisit":    "chw_area",
    "population":   "chw_area",
    "active_chps":  "chw_area_uuid",
}


class DataMerger:
    """
    Constructs the final analytical DataFrame via sequential left-joins.
    All joins are at the chw_area level for contextual features,
    and patient_id level for individual clinical history.
    """

    def build_analytical_dataset(
        self, tables: Dict[str, pd.DataFrame]
    ) -> pd.DataFrame:
        """
        Execute all 10 ETL steps and return the merged analytical DataFrame.
        """
        logger.info("=" * 60)
        logger.info("BUILDING ANALYTICAL DATASET — 10-step ETL pipeline")
        logger.info("=" * 60)

        # Step 1: Deduplicate and prepare iz base table
        df = self._step1_deduplicate_iz(tables["iz"])

        # Step 2: Compute vaccine completeness score (domain logic)
        df = self._step2_vaccine_completeness(df)

        # Step 3: Construct composite target variable
        df = self._step3_build_target(df)

        # Step 4: Join CHW metadata from active_chps
        df = self._step4_join_active_chps(df, tables.get("active_chps"))

        # Step 5: Join CHW supervision quality features
        df = self._step5_join_supervision(df, tables.get("supervision"))

        # Step 6: Join population workload
        df = self._step6_join_population(df, tables.get("population"))

        # Step 7: Join home visit engagement
        df = self._step7_join_homevisit(df, tables.get("homevisit"))

        # Step 8: Join maternal health-seeking from preg_reg
        df = self._step8_join_preg_reg(
            df,
            tables.get("preg_reg"),
            tables.get("preg_reg2"),
        )

        # Step 9: Attach PNC label for audit (NOT a training feature)
        df = self._step9_attach_pnc_audit(df, tables.get("pnc"))

        # Step 10: Final validation and dtype cleanup
        df = self._step10_validate_and_finalize(df)

        logger.info(f"\nAnalytical dataset: {df.shape[0]} rows × {df.shape[1]} cols")
        logger.info(f"Target prevalence:  {df['is_defaulter'].mean():.1%}")
        return df

    # ── Step 1 ─────────────────────────────────────────────────────────────

    def _step1_deduplicate_iz(self, iz: pd.DataFrame) -> pd.DataFrame:
        logger.info("\n[Step 1] Deduplicating iz base table...")
        before = len(iz)

        # Parse reported as datetime if not already
        iz["reported"] = pd.to_datetime(iz["reported"], errors="coerce", utc=True)

        # Standardise month
        if "month_clean" not in iz.columns:
            iz["month_clean"] = iz["month"].copy()

        # Keep one record per patient per month (latest reported)
        iz = (
            iz
            .sort_values("reported", ascending=False)
            .drop_duplicates(subset=["patient_id", "month_clean"], keep="first")
            .reset_index(drop=True)
        )

        logger.info(f"  {before} → {len(iz)} rows (removed {before - len(iz)} duplicates)")
        return iz

    # ── Step 2 ─────────────────────────────────────────────────────────────

    def _step2_vaccine_completeness(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("\n[Step 2] Computing vaccine completeness scores...")

        from config.epi_schedule import CORE_VACCINES, MALARIA_VACCINES, get_expected_vaccines

        # Age-gated expected vaccine count
        df["age_expected_vaccine_count"] = df["patient_age_in_months"].apply(
            lambda age: get_expected_vaccines(age, malaria_endemic=False)
        )
        df["age_expected_vaccine_count_endemic"] = df.apply(
            lambda r: get_expected_vaccines(
                r["patient_age_in_months"],
                malaria_endemic=(r.get("is_in_malaria_endemic_region", "no") == "yes")
            ), axis=1
        )

        # Ensure vaccine columns are numeric 0/1
        all_vax = CORE_VACCINES + MALARIA_VACCINES
        for col in all_vax:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        # Core vaccine completeness (age-gated numerator)
        core_vax_present = [c for c in CORE_VACCINES if c in df.columns]
        df["vax_received_core"] = df[core_vax_present].sum(axis=1)
        df["vax_completeness_score"] = np.where(
            df["age_expected_vaccine_count"] > 0,
            df["vax_received_core"] / df["age_expected_vaccine_count"],
            np.nan
        ).clip(0, 1)

        # All vaccines including malaria
        all_vax_present = [c for c in all_vax if c in df.columns]
        df["vax_received_total"] = df[all_vax_present].sum(axis=1)
        df["vax_completeness_all"] = np.where(
            df["age_expected_vaccine_count_endemic"] > 0,
            df["vax_received_total"] / df["age_expected_vaccine_count_endemic"],
            np.nan
        ).clip(0, 1)

        # Series completion flags (for SHAP interpretability)
        series = {
            "penta_series_complete": ["has_penta_1", "has_penta_2", "has_penta_3"],
            "opv_series_complete":   ["has_opv_1", "has_opv_2", "has_opv_3"],
            "pcv_series_complete":   ["has_pcv_1", "has_pcv_2", "has_pcv_3"],
            "rota_series_complete":  ["has_rota_1", "has_rota_2", "has_rota_3"],
        }
        for flag, cols in series.items():
            present = [c for c in cols if c in df.columns]
            if present:
                df[flag] = (df[present].sum(axis=1) == len(present)).astype(float)

        # Measles booster gap: got MR1 but NOT MR2 and old enough
        if "has_measles_9_months" in df.columns and "has_measles_18_months" in df.columns:
            df["measles_booster_gap"] = np.where(
                (df["has_measles_9_months"] == 1) &
                (df["has_measles_18_months"] == 0) &
                (df["patient_age_in_months"] >= 20),
                1, 0
            ).astype(float)
        else:
            df["measles_booster_gap"] = 0.0

        # Months since last visit
        df["months_since_reported"] = (
            (pd.Timestamp.now(tz="UTC") - df["reported"]).dt.days / 30.44
        ).clip(0, 24)

        logger.info(f"  Completeness score: mean={df['vax_completeness_score'].mean():.3f}")
        return df

    # ── Step 3 ─────────────────────────────────────────────────────────────

    def _step3_build_target(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("\n[Step 3] Constructing composite target variable...")

        needs_fu = df.get("needs_follow_up_binary", df.get("needs_follow_up", "")).copy()
        if needs_fu.dtype == object:
            needs_fu = needs_fu.str.lower().map({"yes": 1, "no": 0})
        needs_fu = pd.to_numeric(needs_fu, errors="coerce").fillna(0)

        due_pos = pd.to_numeric(
            df.get("due_count_clean", df.get("due_count", 0)), errors="coerce"
        ).fillna(0).gt(0).astype(int)

        missed_flag = df.get("vaccines_due_missed", pd.Series("", index=df.index))
        missed_pos = (
            missed_flag.notna() & (missed_flag != "") & (missed_flag.astype(str) != "nan")
        ).astype(int)

        df["is_defaulter"] = ((needs_fu == 1) | (due_pos == 1) | (missed_pos == 1)).astype(int)

        # Component breakdown (useful for analysis)
        df["_target_needs_fu"]    = needs_fu.astype(int)
        df["_target_due_pos"]     = due_pos.astype(int)
        df["_target_missed_flag"] = missed_pos.astype(int)

        pos_rate = df["is_defaulter"].mean()
        logger.info(f"  Positive rate: {pos_rate:.1%} ({df['is_defaulter'].sum()}/{len(df)})")

        if pos_rate < 0.05:
            logger.warning("  Very low positive rate (<5%) — check full dataset extraction")
        if pos_rate > 0.80:
            logger.warning("  Very high positive rate (>80%) — review target definition")

        return df

    # ── Step 4 ─────────────────────────────────────────────────────────────

    def _step4_join_active_chps(
        self, df: pd.DataFrame, chps: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        logger.info("\n[Step 4] Joining CHW metadata from active_chps...")
        if chps is None:
            logger.warning("  active_chps not available — skipping")
            return df

        chps_agg = (
            chps
            .drop_duplicates(subset=["chw_area_uuid"])
            [["chw_area_uuid", "community_unit", "county_name", "sub_county_name"]]
            .rename(columns={"chw_area_uuid": "chw_area_key"})
        )
        df["chw_area_key"] = df["contact_parent_id"]
        df = df.merge(chps_agg, on="chw_area_key", how="left")
        matched = df["community_unit"].notna().sum()
        logger.info(f"  Matched {matched}/{len(df)} rows ({matched/len(df):.0%})")
        return df

    # ── Step 5 ─────────────────────────────────────────────────────────────

    def _step5_join_supervision(
        self, df: pd.DataFrame, sup: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        logger.info("\n[Step 5] Joining CHW supervision quality features...")
        if sup is None:
            logger.warning("  supervision not available — skipping")
            return df

        sup["reported"] = pd.to_datetime(sup["reported"], errors="coerce", utc=True)

        # Aggregate per CHW area: latest supervision + frequency + avg scores
        agg = (
            sup
            .groupby("chw_area")
            .agg(
                chw_supervision_frequency   = ("reported", "count"),
                chw_last_supervision_date   = ("reported", "max"),
                chw_immunization_competency = ("pct_immunization_pct", "mean")
                    if "pct_immunization_pct" in sup.columns
                    else ("calc_immunization_score", "mean"),
                chw_overall_assessment_pct  = ("pct_assessment_pct", "mean")
                    if "pct_assessment_pct" in sup.columns
                    else ("calc_assessment_score", "mean"),
                chw_has_all_tools           = ("has_all_tools_binary", "max")
                    if "has_all_tools_binary" in sup.columns
                    else ("has_all_tools", lambda x: (x == "yes").any().astype(int)),
                chw_has_ppe                 = ("has_proper_protective_equipment_binary", "max")
                    if "has_proper_protective_equipment_binary" in sup.columns
                    else ("has_proper_protective_equipment",
                          lambda x: (x == "yes").any().astype(int)),
            )
            .reset_index()
            .rename(columns={"chw_area": "chw_area_key"})
        )

        # Months since last supervision
        now = pd.Timestamp.now(tz="UTC")
        agg["months_since_last_supervision"] = (
            (now - agg["chw_last_supervision_date"]).dt.days / 30.44
        ).clip(0, 24)

        # Safe rename for immunization competency
        if "chw_immunization_competency" in agg.columns:
            agg["chw_immunization_competency_pct"] = agg["chw_immunization_competency"]

        df = df.merge(agg.drop(columns=["chw_last_supervision_date"], errors="ignore"),
                      on="chw_area_key", how="left")
        matched = df["chw_supervision_frequency"].notna().sum()
        logger.info(f"  Matched {matched}/{len(df)} rows ({matched/len(df):.0%})")
        return df

    # ── Step 6 ─────────────────────────────────────────────────────────────

    def _step6_join_population(
        self, df: pd.DataFrame, pop: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        logger.info("\n[Step 6] Joining population workload (u2_pop)...")
        if pop is None:
            logger.warning("  population not available — skipping")
            return df

        # When loaded from postgres the aggregation is already done in SQL;
        # when loaded from CSV the raw rows need groupby here.
        if "chw_workload_u2" in pop.columns:
            pop_agg = pop[["chw_area", "chw_workload_u2"]].rename(
                columns={"chw_area": "chw_area_key"}
            )
        else:
            pop_agg = (
                pop
                .groupby("chw_area")
                .agg(chw_workload_u2=("u2_pop", "median"))
                .reset_index()
                .rename(columns={"chw_area": "chw_area_key"})
            )
        df = df.merge(pop_agg, on="chw_area_key", how="left")
        matched = df["chw_workload_u2"].notna().sum()
        logger.info(f"  Matched {matched}/{len(df)} rows ({matched/len(df):.0%})")
        return df

    # ── Step 7 ─────────────────────────────────────────────────────────────

    def _step7_join_homevisit(
        self, df: pd.DataFrame, hv: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        logger.info("\n[Step 7] Joining home visit frequency...")
        if hv is None:
            logger.warning("  homevisit not available — skipping")
            return df

        # When loaded from postgres the aggregation is already done in SQL;
        # when loaded from CSV the raw rows need groupby here.
        if "monthly_homevisit_rate" in hv.columns:
            hv_agg = hv[["chw_area", "monthly_homevisit_rate"]].rename(
                columns={"chw_area": "chw_area_key"}
            )
        else:
            hv["reported"] = pd.to_datetime(hv["reported"], errors="coerce", utc=True)
            hv_agg = (
                hv
                .groupby("chw_area")
                .agg(
                    total_homevisits   = ("family_id", "count"),
                    hv_distinct_months = ("reported", lambda x: x.dt.to_period("M").nunique()),
                )
                .reset_index()
            )
            hv_agg["monthly_homevisit_rate"] = (
                hv_agg["total_homevisits"] / hv_agg["hv_distinct_months"].clip(1)
            )
            hv_agg = hv_agg[["chw_area", "monthly_homevisit_rate"]].rename(
                columns={"chw_area": "chw_area_key"}
            )

        df = df.merge(hv_agg, on="chw_area_key", how="left")
        matched = df["monthly_homevisit_rate"].notna().sum()
        logger.info(f"  Matched {matched}/{len(df)} rows ({matched/len(df):.0%})")
        return df

    # ── Step 8 ─────────────────────────────────────────────────────────────

    def _step8_join_preg_reg(
        self,
        df: pd.DataFrame,
        preg_reg: Optional[pd.DataFrame],
        preg_reg2: Optional[pd.DataFrame],
    ) -> pd.DataFrame:
        logger.info("\n[Step 8] Joining maternal health-seeking (preg_reg UNION)...")

        parts = [t for t in [preg_reg, preg_reg2] if t is not None]
        if not parts:
            logger.warning("  No preg_reg tables available — skipping")
            return df

        # Union preg_reg and preg_reg2 on common columns
        common_cols = list(set.intersection(*[set(p.columns) for p in parts]))
        preg = pd.concat([p[common_cols] for p in parts], ignore_index=True)

        # ── Join key logic ────────────────────────────────────────────────────
        # In CHT hierarchy:
        #   iz patient   → contact_parent_id       = CHW area UUID
        #   preg patient → contact_parent_id        = household UUID
        #   preg patient → contact_parent_parent_id = CHW area UUID  ← matches iz
        #
        # A mother→child individual linkage is not available in this dataset.
        # We therefore aggregate maternal features at the CHW AREA level and join
        # on area UUID. This captures "maternal health culture of the catchment area"
        # which is a meaningful contextual predictor of child immunization outcomes.

        # Determine the area-level key in preg_reg
        if "contact_parent_parent_id" in preg.columns:
            area_key_col = "contact_parent_parent_id"
        elif "chw_area" in preg.columns:
            area_key_col = "chw_area"
        else:
            logger.warning(
                "  preg_reg has neither contact_parent_parent_id nor chw_area — "
                "cannot resolve CHW area linkage; maternal features will be null"
            )
            for col in ["maternal_anc_visits", "maternal_anc_defaulter",
                        "maternal_muac_risk", "maternal_iron_folate"]:
                df[col] = np.nan
            return df

        # Build aggregation dict dynamically based on available columns
        agg_spec: dict = {}
        if "number_of_anc_attended" in preg.columns:
            agg_spec["maternal_anc_visits"] = ("number_of_anc_attended", "mean")
        if "is_anc_defaulter_binary" in preg.columns:
            agg_spec["maternal_anc_defaulter"] = ("is_anc_defaulter_binary", "mean")
        elif "is_anc_defaulter" in preg.columns:
            preg["_anc_def"] = (
                preg["is_anc_defaulter"].astype(str).str.lower()
                .isin(["true", "yes", "1"]).astype(float)
            )
            agg_spec["maternal_anc_defaulter"] = ("_anc_def", "mean")
        if "muac_risk" in preg.columns:
            agg_spec["maternal_muac_risk"] = ("muac_risk", "mean")
        elif "muac_color" in preg.columns:
            preg["_muac_risk"] = (~preg["muac_color"].str.lower().eq("green")).astype(float)
            agg_spec["maternal_muac_risk"] = ("_muac_risk", "mean")
        if "iron_folate_binary" in preg.columns:
            agg_spec["maternal_iron_folate"] = ("iron_folate_binary", "mean")
        elif "takes_iron_or_folate_supplements" in preg.columns:
            preg["_fe"] = (
                preg["takes_iron_or_folate_supplements"].str.lower()
                .eq("yes").astype(float)
            )
            agg_spec["maternal_iron_folate"] = ("_fe", "mean")

        if not agg_spec:
            logger.warning("  preg_reg has none of the expected maternal columns — skipping")
            for col in ["maternal_anc_visits", "maternal_anc_defaulter",
                        "maternal_muac_risk", "maternal_iron_folate"]:
                df[col] = np.nan
            return df

        preg_area_agg = (
            preg
            .dropna(subset=[area_key_col])
            .groupby(area_key_col)
            .agg(**agg_spec)
            .reset_index()
            .rename(columns={area_key_col: "chw_area_key"})
        )

        df_join = df.merge(preg_area_agg, on="chw_area_key", how="left")
        matched = df_join["maternal_anc_visits"].notna().sum() if "maternal_anc_visits" in df_join.columns else 0
        logger.info(
            f"  Matched {matched}/{len(df_join)} rows ({matched/len(df_join):.0%}) "
            f"[join: iz.contact_parent_id → preg_reg.{area_key_col}]"
        )
        if matched == 0:
            logger.warning(
                "  0 rows matched. Likely cause: UUID format mismatch between "
                f"iz.contact_parent_id and preg_reg.{area_key_col}. "
                "Inspect both columns for casing/prefix differences."
            )
        return df_join

    # ── Step 9 ─────────────────────────────────────────────────────────────

    def _step9_attach_pnc_audit(
        self, df: pd.DataFrame, pnc: Optional[pd.DataFrame]
    ) -> pd.DataFrame:
        logger.info("\n[Step 9] Attaching PNC label for audit (NOT a feature)...")
        if pnc is None:
            logger.warning("  pnc not available — skipping")
            df["_pnc_iz_defaulter"] = np.nan
            return df

        pnc_labels = (
            pnc
            .dropna(subset=["is_immunization_defaulter_binary"])
            [["patient_id", "is_immunization_defaulter_binary"]]
            .rename(columns={"is_immunization_defaulter_binary": "_pnc_iz_defaulter"})
            .drop_duplicates(subset=["patient_id"])
        )
        df = df.merge(pnc_labels, on="patient_id", how="left")

        # Label agreement rate where both available
        both = df[df["_pnc_iz_defaulter"].notna()]
        if len(both) > 0:
            agreement = (both["is_defaulter"] == both["_pnc_iz_defaulter"]).mean()
            logger.info(f"  Label agreement (iz vs pnc): {agreement:.1%} on {len(both)} matched")
            if agreement < 0.70:
                logger.warning(
                    f"  Low PNC label agreement ({agreement:.1%}). Likely causes:\n"
                    "    1. PNC patient_id may be the MOTHER's ID (not the child's)\n"
                    "       → joined records are incorrect demographic matches\n"
                    "    2. PNC 'is_immunization_defaulter' reflects a different\n"
                    "       assessment window than the iz composite target\n"
                    "    This field is audit-only and NOT a training feature."
                )
        else:
            logger.info("  No matching patient_ids between iz and pnc.")
        return df

    # ── Step 10 ────────────────────────────────────────────────────────────

    def _step10_validate_and_finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("\n[Step 10] Final validation and cleanup...")

        # Encode geographic categoricals
        for col in ["sub_county_name", "county_name", "community_unit", "county"]:
            if col in df.columns:
                enc_col = col.replace("_name", "") + "_encoded"
                df[enc_col] = df[col].astype("category").cat.codes

        # Collapse to one row per patient (keep the most recent record).
        # After Step 1, the data contains one row per (patient_id, month_clean),
        # so patients seen in multiple months appear multiple times.  We sort
        # descending by `reported` first so keep="first" always retains the
        # most recent observation, regardless of join order.
        before = len(df)
        n_patients_before = df["patient_id"].nunique()
        if "reported" in df.columns:
            df = df.sort_values("reported", ascending=False)
        df = df.drop_duplicates(subset=["patient_id"], keep="first").reset_index(drop=True)
        n_removed = before - len(df)
        if n_removed > 0:
            logger.info(
                f"  Collapsed {before} rows → {len(df)} unique patients "
                f"(removed {n_removed} earlier-month records for "
                f"{before - n_patients_before} multi-month patients)"
            )

        # Assertions
        assert "is_defaulter" in df.columns, "Target column missing!"
        assert df["is_defaulter"].isin([0, 1]).all(), "Target has non-binary values!"
        assert df["patient_id"].is_unique, "Duplicate patient_ids in final dataset!"

        pos_rate = df["is_defaulter"].mean()
        if not (0.02 <= pos_rate <= 0.95):
            logger.warning(f"  Unusual positive rate: {pos_rate:.1%}")

        logger.info(f"\n{'='*40}")
        logger.info(f"  Final dataset: {df.shape}")
        logger.info(f"  Defaulters: {df['is_defaulter'].sum()} ({pos_rate:.1%})")
        logger.info(f"  Missing data: {df.isnull().mean().mean():.1%} overall")
        logger.info(f"{'='*40}")
        return df
