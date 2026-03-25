"""
monitoring/drift_detector.py
─────────────────────────────────────────────────────────────────────────────
Production drift detection using Population Stability Index (PSI).
Alerts when feature distributions shift significantly between training
and serving — critical for responsible deployment validation.
"""

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# PSI thresholds (standard industry convention)
PSI_GREEN  = 0.1   # No significant change
PSI_AMBER  = 0.2   # Some change — investigate
PSI_RED    = 0.25  # Significant change — consider retraining


class DriftDetector:
    """
    Monitors feature and prediction distribution drift between
    training baseline and live scoring batches.
    """

    def __init__(self, n_bins: int = 10):
        self.n_bins = n_bins
        self.reference_stats: Optional[Dict] = None

    def fit_reference(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Store reference distribution statistics from training data."""
        self.reference_stats = {}
        skipped = 0
        for col in X_train.select_dtypes(include=[np.number]).columns:
            vals = X_train[col].dropna().values
            if len(vals) < 10:
                skipped += 1
                continue

            # Skip near-constant features (std ≈ 0): e.g. chw_has_all_tools=1.0
            # Percentile binning on a constant vector yields identical edges,
            # which cause np.histogram to misbehave and produce PSI >> 1.
            if np.std(vals) < 1e-8:
                skipped += 1
                continue

            # Skip low-cardinality features (fewer unique values than bins):
            # e.g. county_encoded (3 counties), chw_immunization_competency_pct
            # (9 discrete values). Percentile bins cluster at repeated values,
            # producing degenerate reference proportions and PSI >> 1 even for
            # near-identical distributions.
            n_unique = len(np.unique(vals))
            if n_unique < self.n_bins:
                skipped += 1
                continue

            bins = np.percentile(vals, np.linspace(0, 100, self.n_bins + 1))
            unique_bins = np.unique(bins)
            if len(unique_bins) < 2:
                skipped += 1
                continue

            # Store actual reference bin counts (not assumed uniform).
            # Using unique_bins avoids duplicate-edge errors in np.histogram.
            ref_counts, _ = np.histogram(vals, bins=unique_bins)

            self.reference_stats[col] = {
                "bins":       unique_bins,
                "ref_counts": ref_counts,
                "mean":       float(np.mean(vals)),
                "std":        float(np.std(vals)),
            }
        self.reference_stats["_label_rate"] = float(y_train.mean())
        n_features = len(self.reference_stats) - 1
        logger.info(f"  Reference stats fitted on {len(X_train)} samples, "
                    f"{n_features} numeric features ({skipped} skipped: "
                    f"near-constant or low-cardinality)")

    def detect(
        self,
        X_new: pd.DataFrame,
        y_new: Optional[pd.Series] = None,
        psi_threshold: float = PSI_AMBER,
    ) -> pd.DataFrame:
        """
        Compute PSI for each feature and flag drifted features.

        Returns
        -------
        pd.DataFrame with columns [feature, psi, status, mean_train, mean_new]
        """
        if self.reference_stats is None:
            raise RuntimeError("Call fit_reference() before detect()")

        rows = []
        for col, ref in self.reference_stats.items():
            if col == "_label_rate":
                continue
            if col not in X_new.columns:
                continue

            new_vals = X_new[col].dropna().values
            if len(new_vals) < 5:
                continue

            psi = self._compute_psi(new_vals, ref["bins"], ref["ref_counts"])
            status = (
                "GREEN"  if psi < PSI_GREEN else
                "AMBER"  if psi < PSI_AMBER else
                "RED"
            )
            rows.append({
                "feature":    col,
                "psi":        round(psi, 4),
                "status":     status,
                "mean_train": round(ref["mean"], 4),
                "mean_new":   round(float(np.mean(new_vals)), 4),
                "alert":      psi >= psi_threshold,
            })

        df = pd.DataFrame(rows) if rows else pd.DataFrame(columns=["feature","psi","status","mean_train","mean_new","alert"])
        if not df.empty:
            df = df.sort_values("psi", ascending=False)

        # Label drift (if labels available)
        if y_new is not None and "_label_rate" in self.reference_stats:
            new_rate = float(y_new.mean())
            ref_rate = self.reference_stats["_label_rate"]
            label_shift = abs(new_rate - ref_rate)
            logger.info(f"  Label rate: train={ref_rate:.3f} | new={new_rate:.3f} "
                        f"| shift={label_shift:.3f}")
            if label_shift > 0.10:
                logger.warning("  Label rate shifted >10pp — review ground truth")

        n_alert = df["alert"].sum()
        logger.info(f"\n  Drift report: {n_alert}/{len(df)} features drifted "
                    f"(PSI ≥ {psi_threshold})")
        if n_alert > 0:
            logger.warning(f"  Drifted features:\n{df[df['alert']].to_string(index=False)}")

        return df

    def report_html(self, drift_df: pd.DataFrame) -> str:
        """Generate simple HTML drift report."""
        status_colors = {"GREEN": "#00B050", "AMBER": "#FFC000", "RED": "#C00000"}
        rows_html = ""
        for _, row in drift_df.iterrows():
            color = status_colors.get(row["status"], "#000000")
            rows_html += f"""
            <tr>
              <td>{row['feature']}</td>
              <td>{row['psi']:.4f}</td>
              <td style="color:{color}; font-weight:bold">{row['status']}</td>
              <td>{row['mean_train']:.4f}</td>
              <td>{row['mean_new']:.4f}</td>
            </tr>"""

        return f"""
        <html><body>
        <h2>Feature Drift Report</h2>
        <table border="1" cellpadding="5" cellspacing="0">
          <tr><th>Feature</th><th>PSI</th><th>Status</th>
              <th>Mean (Train)</th><th>Mean (New)</th></tr>
          {rows_html}
        </table>
        <p>PSI: &lt;0.1 GREEN | 0.1–0.2 AMBER | &gt;0.2 RED</p>
        </body></html>"""

    @staticmethod
    def _compute_psi(
        new_vals: np.ndarray,
        ref_bins: np.ndarray,
        ref_counts: np.ndarray,
    ) -> float:
        """
        Population Stability Index (PSI) between reference and new distributions.

        Uses actual reference bin counts (not assumed-uniform) so that PSI is
        correct even when bin edges are not perfectly spaced after deduplication.
        """
        if len(ref_bins) < 2:
            return 0.0

        new_counts, _ = np.histogram(new_vals, bins=ref_bins)

        eps     = 1e-6
        ref_pct = np.clip(ref_counts / max(ref_counts.sum(), 1), eps, None)
        new_pct = np.clip(new_counts / max(new_counts.sum(), 1), eps, None)

        psi = float(np.sum((new_pct - ref_pct) * np.log(new_pct / ref_pct)))
        return max(psi, 0.0)
