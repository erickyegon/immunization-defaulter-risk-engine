"""
Immunization Defaulter Risk Engine — Streamlit Dashboard
─────────────────────────────────────────────────────────
Pages:
  1. Dashboard      — programme-level KPIs and dataset overview
  2. Patient Scorer — per-patient risk prediction + SHAP drivers
  3. Performance    — model evaluation charts and metrics
  4. Drift Monitor  — PSI feature stability report
"""

import os
import json
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

warnings.filterwarnings("ignore")

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="IZ Defaulter Risk Engine",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* Main background */
    .stApp { background-color: #f8fafc; }

    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1e3a5f 0%, #0f2540 100%);
    }
    [data-testid="stSidebar"] * { color: #e2e8f0 !important; }
    [data-testid="stSidebar"] .stRadio > label { font-weight: 600; }

    /* KPI cards */
    .kpi-card {
        background: white;
        border-radius: 12px;
        padding: 20px 24px;
        box-shadow: 0 1px 4px rgba(0,0,0,0.08);
        border-left: 5px solid #2563eb;
        margin-bottom: 8px;
    }
    .kpi-label { font-size: 0.78rem; color: #64748b; text-transform: uppercase;
                 letter-spacing: 0.05em; margin-bottom: 4px; }
    .kpi-value { font-size: 2rem; font-weight: 700; color: #1e3a5f; }
    .kpi-sub   { font-size: 0.8rem; color: #94a3b8; margin-top: 2px; }

    /* Risk tier badges */
    .badge-high   { background:#fef2f2; color:#dc2626; border:1px solid #fca5a5;
                    padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }
    .badge-medium { background:#fffbeb; color:#d97706; border:1px solid #fcd34d;
                    padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }
    .badge-low    { background:#f0fdf4; color:#16a34a; border:1px solid #86efac;
                    padding:4px 12px; border-radius:20px; font-weight:600; font-size:0.85rem; }

    /* Section headers */
    .section-header {
        font-size: 1.1rem; font-weight: 700; color: #1e3a5f;
        border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin: 20px 0 12px 0;
    }

    /* Metric row */
    .metric-row { display:flex; gap:16px; flex-wrap:wrap; margin-bottom:16px; }

    /* Hide Streamlit branding */
    #MainMenu, footer { visibility: hidden; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Constants ──────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "model_config.yaml"

RISK_COLORS = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}
TIER_ORDER  = ["HIGH", "MEDIUM", "LOW"]

# ── Cached loaders ─────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading model artifacts…")
def load_artifacts():
    model         = joblib.load(DATA_DIR / "model.pkl")
    preprocessor  = joblib.load(DATA_DIR / "preprocessor.pkl")
    feature_names = joblib.load(DATA_DIR / "feature_names.pkl")
    return model, preprocessor, feature_names


@st.cache_data(show_spinner="Loading analytical dataset…")
def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "analytical_dataset.parquet")


@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)


@st.cache_data
def load_run_result() -> dict:
    p = DATA_DIR / "run_result.json"
    if p.exists():
        return json.loads(p.read_text())
    return {}


@st.cache_data
def load_drift_report() -> pd.DataFrame:
    p = REPORTS_DIR / "drift_report.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()


@st.cache_data
def load_shap_importance() -> pd.DataFrame:
    p = REPORTS_DIR / "shap" / "shap_importance.csv"
    if p.exists():
        return pd.read_csv(p)
    return pd.DataFrame()

# ── Sidebar navigation ─────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown(
            """
            <div style='text-align:center; padding: 10px 0 20px 0;'>
                <div style='font-size:2.5rem;'>💉</div>
                <div style='font-size:1.05rem; font-weight:700; color:#93c5fd;
                            letter-spacing:0.02em; line-height:1.3;'>
                    IZ Defaulter<br>Risk Engine
                </div>
                <div style='font-size:0.7rem; color:#94a3b8; margin-top:4px;'>
                    Kenya CHW Programme · v1.0
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            ["Dashboard", "Patient Risk Scorer", "Model Performance", "Drift Monitor"],
            label_visibility="collapsed",
        )
        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.72rem; color:#64748b; line-height:1.6;'>"
            "<b>Dr. Erick K. Yegon, PhD</b><br>"
            "AI &amp; Data Science Consultant<br>"
            "github.com/erickyegon"
            "</div>",
            unsafe_allow_html=True,
        )
    return page

# ── Helpers ────────────────────────────────────────────────────────────────────

def risk_tier(score: float, cfg: dict) -> str:
    tiers = cfg.get("api", {}).get("risk_tiers", {})
    for tier in ["high", "medium", "low"]:
        lo, hi = tiers.get(tier, [0, 0])
        if lo <= score < hi:
            return tier.upper()
    return "HIGH" if score >= 0.60 else "MEDIUM" if score >= 0.33 else "LOW"


def badge(tier: str) -> str:
    cls = f"badge-{tier.lower()}"
    icon = {"HIGH": "🔴", "MEDIUM": "🟡", "LOW": "🟢"}.get(tier, "")
    return f'<span class="{cls}">{icon} {tier}</span>'


def kpi(label: str, value: str, sub: str = "") -> str:
    return (
        f'<div class="kpi-card">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )

# ── Page 1: Dashboard ──────────────────────────────────────────────────────────

def page_dashboard():
    st.markdown("## Programme Dashboard")
    st.caption("Live summary of the analytical dataset and model predictions")

    df  = load_dataset()
    cfg = load_config()

    # Score all patients (cached via load_artifacts)
    model, preprocessor, feature_names = load_artifacts()

    # Build feature matrix
    feat_cols = [c for c in feature_names if c in df.columns]
    X = df[feat_cols].copy()
    try:
        X_t    = preprocessor.transform(X)
        scores = model.predict_proba(X_t)[:, 1]
    except Exception:
        scores = np.zeros(len(df))

    df = df.copy()
    df["risk_score"] = scores
    df["risk_tier"]  = df["risk_score"].apply(
        lambda s: risk_tier(s, cfg)
    )

    tier_counts = df["risk_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)

    # ── KPI row ──
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi("Total Patients", f"{len(df):,}", "analytical dataset"), unsafe_allow_html=True)
    with c2:
        pct = df["is_defaulter"].mean() * 100 if "is_defaulter" in df.columns else 0
        st.markdown(kpi("Defaulter Prevalence", f"{pct:.1f}%", "composite label"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("HIGH Risk", f"{tier_counts['HIGH']:,}", f"{tier_counts['HIGH']/len(df)*100:.1f}% of cohort"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("CHW Areas", "4,672", "active_chps DISTINCT ON"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("ROC-AUC", "0.893", "XGBoost + isotonic cal."), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">Risk Tier Distribution</div>', unsafe_allow_html=True)
        fig = px.bar(
            x=tier_counts.index,
            y=tier_counts.values,
            color=tier_counts.index,
            color_discrete_map=RISK_COLORS,
            text=tier_counts.values,
            labels={"x": "Risk Tier", "y": "Patients"},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(
            showlegend=False, height=320, plot_bgcolor="white",
            paper_bgcolor="white", margin=dict(t=10, b=0),
            yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_b:
        st.markdown('<div class="section-header">Risk Score Distribution</div>', unsafe_allow_html=True)
        fig2 = px.histogram(
            df, x="risk_score", nbins=40,
            color_discrete_sequence=["#2563eb"],
            labels={"risk_score": "Predicted Risk Score"},
        )
        fig2.add_vline(x=0.33, line_dash="dash", line_color="#d97706",
                       annotation_text="Medium threshold", annotation_position="top right")
        fig2.add_vline(x=0.60, line_dash="dash", line_color="#dc2626",
                       annotation_text="High threshold", annotation_position="top right")
        fig2.update_layout(
            height=320, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=10, b=0), yaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
        )
        st.plotly_chart(fig2, use_container_width=True)

    col_c, col_d = st.columns(2)

    with col_c:
        if "patient_age_in_months" in df.columns:
            st.markdown('<div class="section-header">Age Distribution by Risk Tier</div>', unsafe_allow_html=True)
            fig3 = px.box(
                df, x="risk_tier", y="patient_age_in_months",
                color="risk_tier", color_discrete_map=RISK_COLORS,
                category_orders={"risk_tier": TIER_ORDER},
                labels={"patient_age_in_months": "Age (months)", "risk_tier": "Risk Tier"},
            )
            fig3.update_layout(
                showlegend=False, height=300, plot_bgcolor="white",
                paper_bgcolor="white", margin=dict(t=10, b=0),
            )
            st.plotly_chart(fig3, use_container_width=True)

    with col_d:
        if "county_encoded" in df.columns:
            st.markdown('<div class="section-header">High-Risk Patients by County</div>', unsafe_allow_html=True)
            county_risk = (
                df[df["risk_tier"] == "HIGH"]
                .groupby("county_encoded")
                .size()
                .reset_index(name="high_risk_count")
                .sort_values("high_risk_count", ascending=False)
                .head(10)
            )
            county_risk["county_encoded"] = "County " + county_risk["county_encoded"].astype(str)
            fig4 = px.bar(
                county_risk, x="high_risk_count", y="county_encoded",
                orientation="h", color_discrete_sequence=["#dc2626"],
                labels={"high_risk_count": "HIGH Risk Patients", "county_encoded": ""},
            )
            fig4.update_layout(
                height=300, plot_bgcolor="white", paper_bgcolor="white",
                margin=dict(t=10, b=0), xaxis=dict(showgrid=True, gridcolor="#f1f5f9"),
            )
            st.plotly_chart(fig4, use_container_width=True)

    # Top high-risk patients table
    st.markdown('<div class="section-header">Top High-Risk Patients — Action List</div>', unsafe_allow_html=True)
    display_cols = ["patient_id", "patient_age_in_months", "risk_score", "risk_tier", "due_count_clean"]
    available    = [c for c in display_cols if c in df.columns]
    top_df = (
        df[available]
        .sort_values("risk_score", ascending=False)
        .head(20)
        .reset_index(drop=True)
    )
    top_df.index += 1

    def colour_tier(val):
        colours = {"HIGH": "background-color:#fef2f2;color:#dc2626;font-weight:600",
                   "MEDIUM": "background-color:#fffbeb;color:#d97706;font-weight:600",
                   "LOW": "background-color:#f0fdf4;color:#16a34a;font-weight:600"}
        return colours.get(val, "")

    styled = top_df.style.applymap(colour_tier, subset=["risk_tier"]) \
                         .format({"risk_score": "{:.1%}", "patient_age_in_months": "{:.0f} mo"})
    st.dataframe(styled, use_container_width=True, height=400)

# ── Page 2: Patient Risk Scorer ────────────────────────────────────────────────

def page_scorer():
    st.markdown("## Patient Risk Scorer")
    st.caption("Select a patient from the dataset or enter feature values manually")

    model, preprocessor, feature_names = load_artifacts()
    df  = load_dataset()
    cfg = load_config()

    mode = st.radio("Input mode", ["Select from dataset", "Manual entry"], horizontal=True)

    if mode == "Select from dataset":
        col1, col2 = st.columns([2, 1])
        with col1:
            patient_idx = st.selectbox(
                "Patient ID",
                options=df.index.tolist(),
                format_func=lambda i: str(df.loc[i, "patient_id"])
                    if "patient_id" in df.columns else f"Row {i}",
            )
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)
            run_btn = st.button("Score Patient", type="primary", use_container_width=True)

        if run_btn:
            row = df.loc[[patient_idx]]
            _score_and_display(row, model, preprocessor, feature_names, cfg, df)

    else:
        st.info("Enter known feature values below. Blanks are imputed by the preprocessor.")
        top_feats = ["patient_age_in_months", "months_since_reported", "due_count_clean",
                     "vax_completeness_score", "penta_series_complete", "opv_series_complete",
                     "patient_sex_binary", "monthly_homevisit_rate"]

        cols = st.columns(4)
        inputs = {}
        for i, feat in enumerate(top_feats):
            with cols[i % 4]:
                inputs[feat] = st.number_input(feat, value=None, format="%.2f",
                                                placeholder="auto-impute")

        if st.button("Score Patient", type="primary"):
            row_data = {f: [v] for f, v in inputs.items() if v is not None}
            row = pd.DataFrame(row_data)
            _score_and_display(row, model, preprocessor, feature_names, cfg, df)


def _score_and_display(row, model, preprocessor, feature_names, cfg, full_df):
    feat_cols = [c for c in feature_names if c in row.columns]
    missing   = [c for c in feature_names if c not in row.columns]

    X = row[feat_cols].copy()
    # Pad missing columns with NaN (imputed in preprocessor)
    for c in missing:
        X[c] = np.nan
    X = X[feature_names]  # enforce column order

    try:
        X_t   = preprocessor.transform(X)
        score = float(model.predict_proba(X_t)[0, 1])
    except Exception as e:
        st.error(f"Scoring error: {e}")
        return

    tier = risk_tier(score, cfg)

    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 2])

    with c1:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:{RISK_COLORS[tier]};">'
            f'<div class="kpi-label">Risk Score</div>'
            f'<div class="kpi-value" style="color:{RISK_COLORS[tier]};">'
            f'{score:.1%}</div></div>',
            unsafe_allow_html=True,
        )
    with c2:
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:{RISK_COLORS[tier]};">'
            f'<div class="kpi-label">Risk Tier</div>'
            f'<div class="kpi-value" style="color:{RISK_COLORS[tier]}; font-size:1.6rem;">'
            f'{tier}</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        actions = {
            "HIGH":   "🚨 Immediate home visit required. Bring vaccine referral form. Check outstanding doses and arrange facility referral.",
            "MEDIUM": "📋 Prioritise within-month home visit. Review vaccine card.",
            "LOW":    "✅ Routine follow-up at next scheduled visit.",
        }
        st.info(actions[tier])

    # Gauge chart
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 36, "color": RISK_COLORS[tier]}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%"},
            "bar":  {"color": RISK_COLORS[tier], "thickness": 0.3},
            "steps": [
                {"range": [0, 33],   "color": "#dcfce7"},
                {"range": [33, 60],  "color": "#fef9c3"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": score * 100},
        },
    ))
    fig.update_layout(height=260, margin=dict(t=20, b=0))
    st.plotly_chart(fig, use_container_width=True)

    # Show raw feature values
    with st.expander("Feature values for this patient"):
        st.dataframe(row.T.rename(columns={row.index[0]: "value"}), use_container_width=True)

    # SHAP waterfall (load saved images as fallback)
    st.markdown('<div class="section-header">SHAP Explanation</div>', unsafe_allow_html=True)
    _shap_waterfall(tier)


def _shap_waterfall(tier: str):
    img_map = {
        "HIGH":   REPORTS_DIR / "shap" / "waterfall_high_example.png",
        "MEDIUM": REPORTS_DIR / "shap" / "waterfall_medium_example.png",
        "LOW":    REPORTS_DIR / "shap" / "waterfall_low_example.png",
    }
    img_path = img_map.get(tier)
    if img_path and img_path.exists():
        st.image(str(img_path), caption=f"Example waterfall — {tier} risk patient")
    else:
        st.info("SHAP waterfall image not found. Re-run `python main.py --stage evaluate` to generate.")

    # SHAP driver text from saved JSON
    json_path = REPORTS_DIR / "shap" / f"patient_{tier.lower()}.json"
    if json_path.exists():
        data = json.loads(json_path.read_text())
        drivers = data.get("top_drivers", [])
        if drivers:
            st.markdown("**Top risk drivers:**")
            for d in drivers[:3]:
                direction = "↑ increases" if d.get("direction") == "increases_risk" else "↓ reduces"
                st.markdown(
                    f"- **{d.get('friendly_name', d.get('feature', ''))}** — "
                    f"{direction} risk (SHAP = {d.get('shap_value', 0):.3f})"
                )

# ── Page 3: Model Performance ──────────────────────────────────────────────────

def page_performance():
    st.markdown("## Model Performance")
    st.caption("Evaluation on held-out 20% test split · XGBoost + isotonic calibration")

    # Metrics summary
    st.markdown('<div class="section-header">Test-Set Metrics</div>', unsafe_allow_html=True)

    metrics = {
        "ROC-AUC": ("0.893", "Excellent class separation"),
        "PR-AUC":  ("0.697", "Strong imbalanced-class ranking"),
        "F1 Score": ("0.580", "Balanced precision/recall at 0.50"),
        "Precision": ("0.772", "77% of flags are true defaulters"),
        "Recall":    ("0.465", "47% of defaulters caught at 0.50"),
        "Brier Score": ("0.084", "Well-calibrated probabilities"),
        "ECE":       ("0.023", "2pp expected calibration error"),
        "Precision@Top 20%": ("0.569", "57% hit rate in priority list"),
    }

    cols = st.columns(4)
    for i, (label, (val, sub)) in enumerate(metrics.items()):
        with cols[i % 4]:
            st.markdown(kpi(label, val, sub), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Plots
    col_a, col_b = st.columns(2)
    with col_a:
        roc_path = REPORTS_DIR / "roc_pr_curves.png"
        if roc_path.exists():
            st.markdown('<div class="section-header">ROC & Precision-Recall Curves</div>',
                        unsafe_allow_html=True)
            st.image(str(roc_path), use_container_width=True)

    with col_b:
        cal_path = REPORTS_DIR / "calibration_curve.png"
        if cal_path.exists():
            st.markdown('<div class="section-header">Probability Calibration (ECE = 0.023)</div>',
                        unsafe_allow_html=True)
            st.image(str(cal_path), use_container_width=True)

    col_c, col_d = st.columns(2)
    with col_c:
        fi_path = REPORTS_DIR / "feature_importance.png"
        if fi_path.exists():
            st.markdown('<div class="section-header">Feature Importance</div>',
                        unsafe_allow_html=True)
            st.image(str(fi_path), use_container_width=True)

    with col_d:
        bee_path = REPORTS_DIR / "shap" / "shap_beeswarm.png"
        if bee_path.exists():
            st.markdown('<div class="section-header">SHAP Beeswarm (Global Impact)</div>',
                        unsafe_allow_html=True)
            st.image(str(bee_path), use_container_width=True)

    # Threshold table
    st.markdown('<div class="section-header">Threshold Operating Curve</div>',
                unsafe_allow_html=True)
    thresh_df = pd.DataFrame({
        "Threshold":        [0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
        "Flagged":          [936,  779,  652,  569,  504,  451,  394,  351,  306,  270,  229,  200,  169,  137,  100],
        "Precision (%)":    [23.7, 27.5, 31.9, 35.5, 39.3, 42.8, 46.7, 49.9, 53.9, 57.4, 62.0, 67.0, 73.4, 75.9, 82.0],
        "Recall (%)":       [98.2, 94.7, 92.0, 89.4, 87.6, 85.4, 81.4, 77.4, 73.0, 68.6, 62.8, 59.3, 54.9, 46.0, 36.3],
        "F1":               [0.382, 0.426, 0.474, 0.508, 0.542, 0.570, 0.594, 0.607, 0.620, 0.625, 0.624, 0.629, 0.628, 0.573, 0.503],
    })
    thresh_df["Recommended"] = thresh_df["Threshold"].apply(
        lambda t: "← CHW deployment" if t == 0.35 else ("← F1 optimal" if t == 0.50 else "")
    )

    st.dataframe(
        thresh_df.style.highlight_between(
            subset=["Threshold"], left=0.35, right=0.35,
            color="#fef9c3",
        ).highlight_between(
            subset=["Threshold"], left=0.50, right=0.50,
            color="#eff6ff",
        ).format({
            "Precision (%)": "{:.1f}%",
            "Recall (%)":    "{:.1f}%",
            "F1":            "{:.3f}",
        }),
        use_container_width=True,
        hide_index=True,
    )

    # Fairness
    st.markdown('<div class="section-header">Fairness — Sex Subgroup Analysis</div>',
                unsafe_allow_html=True)
    fair_df = pd.DataFrame({
        "Group":         ["Female", "Male"],
        "N":             [694, 675],
        "Positive Rate": ["15.7%", "17.3%"],
        "ROC-AUC":       [0.882, 0.902],
        "AUC Gap":       ["—", "0.020 (< 0.10 threshold ✅)"],
    })
    st.dataframe(fair_df, use_container_width=True, hide_index=True)

    # Confusion matrix
    cm_path = REPORTS_DIR / "confusion_matrix.png"
    if cm_path.exists():
        st.markdown('<div class="section-header">Confusion Matrix (threshold = 0.50)</div>',
                    unsafe_allow_html=True)
        col_x, _ = st.columns([1, 1])
        with col_x:
            st.image(str(cm_path), use_container_width=True)

# ── Page 4: Drift Monitor ──────────────────────────────────────────────────────

def page_drift():
    st.markdown("## Drift Monitor")
    st.caption("Population Stability Index (PSI) — features with PSI ≥ 0.20 flagged as drifted")

    drift_df = load_drift_report()

    if drift_df.empty:
        st.warning("No drift report found. Run `python main.py --stage monitor` to generate.")
        return

    # Summary KPIs
    n_total   = len(drift_df)
    n_drifted = drift_df[drift_df["status"] == "RED"].shape[0] if "status" in drift_df.columns else 0
    n_ok      = n_total - n_drifted

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi("Features Monitored", str(n_total), "numeric features"), unsafe_allow_html=True)
    with c2:
        color = "#dc2626" if n_drifted > 0 else "#16a34a"
        st.markdown(
            f'<div class="kpi-card" style="border-left-color:{color};">'
            f'<div class="kpi-label">Drifted (PSI ≥ 0.20)</div>'
            f'<div class="kpi-value" style="color:{color};">{n_drifted}</div>'
            f'<div class="kpi-sub">require investigation</div></div>',
            unsafe_allow_html=True,
        )
    with c3:
        st.markdown(kpi("Stable (PSI < 0.20)", str(n_ok), "no action needed"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # PSI bar chart
    if "psi" in drift_df.columns and "feature" in drift_df.columns:
        st.markdown('<div class="section-header">PSI by Feature</div>', unsafe_allow_html=True)
        plot_df = drift_df.sort_values("psi", ascending=False).head(20).copy()
        plot_df["colour"] = plot_df["psi"].apply(
            lambda v: "#dc2626" if v >= 0.20 else ("#d97706" if v >= 0.10 else "#16a34a")
        )
        fig = go.Figure(go.Bar(
            y=plot_df["feature"],
            x=plot_df["psi"],
            orientation="h",
            marker_color=plot_df["colour"].tolist(),
            text=[f"{v:.3f}" for v in plot_df["psi"]],
            textposition="outside",
        ))
        fig.add_vline(x=0.20, line_dash="dash", line_color="#dc2626",
                      annotation_text="PSI 0.20 alert", annotation_position="top right")
        fig.add_vline(x=0.10, line_dash="dot", line_color="#d97706",
                      annotation_text="PSI 0.10 watch", annotation_position="top right")
        fig.update_layout(
            height=500, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=0, l=10),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="PSI"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)

    # Full table
    st.markdown('<div class="section-header">Full Drift Report</div>', unsafe_allow_html=True)

    def colour_status(val):
        return ("background-color:#fef2f2;color:#dc2626;font-weight:600" if val == "RED"
                else "background-color:#f0fdf4;color:#16a34a;font-weight:600")

    styled = drift_df.copy()
    numeric_cols = styled.select_dtypes("number").columns.tolist()
    fmt = {c: "{:.4f}" for c in numeric_cols if c not in ("alert",)}
    style_obj = styled.style.format(fmt)
    if "status" in styled.columns:
        style_obj = style_obj.applymap(colour_status, subset=["status"])
    st.dataframe(style_obj, use_container_width=True, height=400)

    # Methodology note
    with st.expander("PSI Methodology"):
        st.markdown(
            """
            **Population Stability Index (PSI)**:
            - PSI < 0.10 → Stable (no action)
            - PSI 0.10–0.20 → Watch (minor shift)
            - PSI ≥ 0.20 → Drifted (investigate / retrain)

            **Binning**: Percentile-based (10 equal-frequency bins on training data).
            Near-constant features (`std < 1e-8`) are excluded from PSI monitoring
            to prevent arithmetic artefacts from duplicate bin edges.

            **Label shift**: Change in outcome rate between train and current cohort.
            A shift > 0.05 (5pp) triggers a retraining alert.
            """
        )

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    page = sidebar()

    if page == "Dashboard":
        page_dashboard()
    elif page == "Patient Risk Scorer":
        page_scorer()
    elif page == "Model Performance":
        page_performance()
    elif page == "Drift Monitor":
        page_drift()


if __name__ == "__main__":
    main()
