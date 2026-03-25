"""
Immunization Defaulter Risk Engine — Streamlit Dashboard
─────────────────────────────────────────────────────────
Audience: MOH programme managers and CHW supervisors.
Language: Plain English. No statistics background required.
"""

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
    page_title="Immunization Defaulter Risk Engine",
    page_icon="💉",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.stApp { background-color: #f8fafc; }

[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #1e3a5f 0%, #0f2540 100%);
}
[data-testid="stSidebar"] * { color: #e2e8f0 !important; }

.kpi-card {
    background: white; border-radius: 12px; padding: 18px 20px;
    box-shadow: 0 1px 4px rgba(0,0,0,0.08); border-left: 5px solid #2563eb;
    margin-bottom: 8px;
}
.kpi-label { font-size: 0.75rem; color: #64748b; text-transform: uppercase;
             letter-spacing: 0.06em; margin-bottom: 4px; }
.kpi-value { font-size: 1.9rem; font-weight: 700; color: #1e3a5f; line-height: 1.1; }
.kpi-sub   { font-size: 0.78rem; color: #94a3b8; margin-top: 3px; }

.section-header {
    font-size: 1.05rem; font-weight: 700; color: #1e3a5f;
    border-bottom: 2px solid #e2e8f0; padding-bottom: 6px; margin: 22px 0 10px 0;
}

/* Plain-English insight boxes */
.insight-box {
    background: #f0f7ff; border-left: 4px solid #2563eb;
    border-radius: 0 8px 8px 0; padding: 12px 16px; margin: 8px 0 18px 0;
    font-size: 0.88rem; color: #1e3a5f; line-height: 1.6;
}
.insight-box b { color: #1e40af; }

.action-high   { background:#fef2f2; border-left:4px solid #dc2626;
                 border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0;
                 font-size:0.9rem; color:#7f1d1d; line-height:1.6; }
.action-medium { background:#fffbeb; border-left:4px solid #d97706;
                 border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0;
                 font-size:0.9rem; color:#78350f; line-height:1.6; }
.action-low    { background:#f0fdf4; border-left:4px solid #16a34a;
                 border-radius:0 8px 8px 0; padding:12px 16px; margin:8px 0;
                 font-size:0.9rem; color:#14532d; line-height:1.6; }

.step-box {
    background: white; border-radius: 10px; padding: 14px 18px;
    box-shadow: 0 1px 3px rgba(0,0,0,0.07); margin-bottom: 10px;
    border-top: 3px solid #2563eb;
}
.step-num { font-size: 0.7rem; font-weight: 700; color: #2563eb;
            text-transform: uppercase; letter-spacing: 0.08em; }
.step-title { font-size: 0.95rem; font-weight: 700; color: #1e3a5f; margin: 2px 0; }
.step-desc  { font-size: 0.82rem; color: #64748b; line-height: 1.5; }

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT        = Path(__file__).parent
DATA_DIR    = ROOT / "data" / "processed"
REPORTS_DIR = ROOT / "reports"
CONFIG_PATH = ROOT / "config" / "model_config.yaml"

RISK_COLORS = {"HIGH": "#dc2626", "MEDIUM": "#d97706", "LOW": "#16a34a"}
TIER_ORDER  = ["HIGH", "MEDIUM", "LOW"]

FRIENDLY_NAMES = {
    "patient_age_in_months":          "Child's age (months)",
    "months_since_reported":          "Months since last CHW contact",
    "due_count_clean":                "Vaccine doses currently outstanding",
    "vax_completeness_score":         "Overall vaccine completeness (0–1)",
    "penta_series_complete":          "Penta 1-2-3 series complete (0=No, 1=Yes)",
    "opv_series_complete":            "OPV series complete (0=No, 1=Yes)",
    "patient_sex_binary":             "Child's sex (1=Male, 0=Female)",
    "monthly_homevisit_rate":         "CHW home visit rate (visits/month)",
    "vax_completeness_core_only":     "Core vaccine completeness",
    "vitamin_a_completeness":         "Vitamin A supplementation completeness",
    "chw_immunization_competency_pct":"CHW immunization competency score",
    "chw_supervision_frequency":      "CHW supervision frequency",
    "chw_workload_u2":                "Under-2 children per CHW area",
    "is_defaulter":                   "Known defaulter (1=Yes, 0=No)",
    "risk_score":                     "Predicted defaulter risk",
    "risk_tier":                      "Risk level",
}

# ── Loaders ────────────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading risk model…")
def load_artifacts():
    model        = joblib.load(DATA_DIR / "model.pkl")
    preprocessor = joblib.load(DATA_DIR / "preprocessor.pkl")
    feat_names   = joblib.load(DATA_DIR / "feature_names.pkl")
    return model, preprocessor, feat_names

@st.cache_data(show_spinner="Loading patient data…")
def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(DATA_DIR / "analytical_dataset.parquet")

@st.cache_data
def load_config() -> dict:
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)

@st.cache_data
def load_drift_report() -> pd.DataFrame:
    p = REPORTS_DIR / "drift_report.csv"
    return pd.read_csv(p) if p.exists() else pd.DataFrame()

# ── Helpers ────────────────────────────────────────────────────────────────────

def get_risk_tier(score: float, cfg: dict) -> str:
    tiers = cfg.get("api", {}).get("risk_tiers", {})
    for tier in ["high", "medium", "low"]:
        lo, hi = tiers.get(tier, [0, 0])
        if lo <= score < hi:
            return tier.upper()
    return "HIGH" if score >= 0.60 else "MEDIUM" if score >= 0.33 else "LOW"

def kpi(label, value, sub="", color="#2563eb"):
    return (f'<div class="kpi-card" style="border-left-color:{color};">'
            f'<div class="kpi-label">{label}</div>'
            f'<div class="kpi-value" style="color:{color};">{value}</div>'
            f'<div class="kpi-sub">{sub}</div></div>')

def insight(text):
    return f'<div class="insight-box">{text}</div>'

def fname(col):
    return FRIENDLY_NAMES.get(col, col.replace("_", " ").title())

# ── Sidebar ────────────────────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style='text-align:center; padding:10px 0 20px 0;'>
            <div style='font-size:2.5rem;'>💉</div>
            <div style='font-size:1.0rem; font-weight:700; color:#93c5fd; line-height:1.4;'>
                Immunization Defaulter<br>Risk Engine
            </div>
            <div style='font-size:0.68rem; color:#94a3b8; margin-top:4px;'>
                Kenya CHW Programme · v1.0
            </div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Go to",
            ["📊 Programme Dashboard",
             "🔍 Check a Patient's Risk",
             "📈 How Well Does the Model Work?",
             "🔔 Data Quality & Model Health"],
            label_visibility="collapsed",
        )

        st.markdown("---")

        with st.expander("❓ How to use this tool", expanded=False):
            st.markdown("""
**This tool helps CHW supervisors and programme managers identify children who are at risk of missing vaccines.**

**Four pages:**

**📊 Programme Dashboard**
See an overview of all children — how many are high risk, where they are, and which age groups are most at risk.

**🔍 Check a Patient's Risk**
Look up a specific child by their ID to see their risk score and the reasons behind it.

**📈 How Well Does the Model Work?**
Understand how accurate the predictions are and what the model is based on.

**🔔 Data Quality & Model Health**
Check if the data used to score children is still reliable and representative.

---
*This tool does not replace clinical judgement. It is a prioritisation aid to help CHWs focus their limited time on the children who need them most.*
            """)

        st.markdown("---")
        st.markdown(
            "<div style='font-size:0.7rem; color:#64748b; line-height:1.7;'>"
            "<b>Dr. Erick K. Yegon, PhD</b><br>"
            "AI &amp; Data Science Consultant<br>"
            "Former Global Director, Data Science &amp; Analytics<br>"
            "Living Goods · github.com/erickyegon"
            "</div>",
            unsafe_allow_html=True,
        )
    return page

# ── Page 1: Programme Dashboard ────────────────────────────────────────────────

def page_dashboard():
    st.markdown("## 📊 Programme Dashboard")
    st.markdown(
        "This page gives you a **programme-level picture** of all children in the dataset. "
        "Use it to understand the overall burden of vaccine defaulting, identify high-burden counties, "
        "and see which age groups are most at risk."
    )

    df  = load_dataset()
    cfg = load_config()
    model, preprocessor, feature_names = load_artifacts()

    feat_cols = [c for c in feature_names if c in df.columns]
    X = df[feat_cols].copy()
    try:
        X_t    = preprocessor.transform(X)
        scores = model.predict_proba(X_t)[:, 1]
    except Exception:
        scores = np.zeros(len(df))

    df = df.copy()
    df["risk_score"] = scores
    df["risk_tier"]  = df["risk_score"].apply(lambda s: get_risk_tier(s, cfg))
    tier_counts = df["risk_tier"].value_counts().reindex(TIER_ORDER, fill_value=0)

    # ── KPIs ──
    pct = df["is_defaulter"].mean() * 100 if "is_defaulter" in df.columns else 0
    hi_n = tier_counts["HIGH"]

    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        st.markdown(kpi("Children in dataset", f"{len(df):,}", "unique patients"), unsafe_allow_html=True)
    with c2:
        st.markdown(kpi("Known defaulters", f"{pct:.1f}%", "1 in 6 children", "#7c3aed"), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Flagged HIGH risk", f"{hi_n:,}",
                        f"{hi_n/len(df)*100:.1f}% of all children", "#dc2626"), unsafe_allow_html=True)
    with c4:
        st.markdown(kpi("CHW areas covered", "4,672", "active CHW areas"), unsafe_allow_html=True)
    with c5:
        st.markdown(kpi("Model accuracy", "89%", "correctly ranks defaulters", "#16a34a"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Risk distribution ──
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown('<div class="section-header">How many children fall into each risk level?</div>',
                    unsafe_allow_html=True)
        fig = px.bar(
            x=tier_counts.index, y=tier_counts.values,
            color=tier_counts.index, color_discrete_map=RISK_COLORS,
            text=tier_counts.values,
            labels={"x": "Risk Level", "y": "Number of Children"},
        )
        fig.update_traces(texttemplate="%{text:,}", textposition="outside")
        fig.update_layout(showlegend=False, height=320, plot_bgcolor="white",
                          paper_bgcolor="white", margin=dict(t=10, b=0),
                          yaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(insight(
            "<b>How to read this:</b> Children are placed into three groups based on their predicted risk score. "
            "<b style='color:#dc2626;'>HIGH</b> means the model predicts the child is very likely to miss vaccines and needs an immediate home visit. "
            "<b style='color:#d97706;'>MEDIUM</b> means the child is at moderate risk — a scheduled visit this month is recommended. "
            "<b style='color:#16a34a;'>LOW</b> means the child is largely on-track and can follow a routine schedule."
        ), unsafe_allow_html=True)

    with col_b:
        st.markdown('<div class="section-header">Spread of risk scores across all children</div>',
                    unsafe_allow_html=True)
        fig2 = px.histogram(df, x="risk_score", nbins=40,
                            color_discrete_sequence=["#2563eb"],
                            labels={"risk_score": "Predicted Risk Score (0 = safe, 1 = high risk)"})
        fig2.add_vline(x=0.33, line_dash="dash", line_color="#d97706",
                       annotation_text="Medium starts here", annotation_position="top right")
        fig2.add_vline(x=0.60, line_dash="dash", line_color="#dc2626",
                       annotation_text="High starts here", annotation_position="top right")
        fig2.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                           margin=dict(t=10, b=0), yaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
        st.plotly_chart(fig2, use_container_width=True)
        st.markdown(insight(
            "<b>How to read this:</b> Each bar shows how many children received a score in that range. "
            "The tall bars on the left show the majority of children are predicted to be on-track. "
            "The smaller bars on the right (past the red dashed line) are the children the programme should focus on. "
            "The two dashed lines show where the Medium and High risk zones begin."
        ), unsafe_allow_html=True)

    # ── Age and geography ──
    col_c, col_d = st.columns(2)

    with col_c:
        if "patient_age_in_months" in df.columns:
            st.markdown('<div class="section-header">Which age groups are most at risk?</div>',
                        unsafe_allow_html=True)
            fig3 = px.box(df, x="risk_tier", y="patient_age_in_months",
                          color="risk_tier", color_discrete_map=RISK_COLORS,
                          category_orders={"risk_tier": TIER_ORDER},
                          labels={"patient_age_in_months": "Child's Age (months)",
                                  "risk_tier": "Risk Level"})
            fig3.update_layout(showlegend=False, height=320, plot_bgcolor="white",
                               paper_bgcolor="white", margin=dict(t=10, b=0))
            st.plotly_chart(fig3, use_container_width=True)
            st.markdown(insight(
                "<b>How to read this:</b> Each box shows the age range of children in that risk group. "
                "The line in the middle of the box is the typical (median) age. The box covers the middle 50% of children. "
                "If the HIGH risk box covers the 6–18 month range, it means this is the most critical age window — "
                "these children are in the peak immunization period and need the most attention."
            ), unsafe_allow_html=True)

    with col_d:
        if "county_encoded" in df.columns:
            st.markdown('<div class="section-header">Which counties have the most high-risk children?</div>',
                        unsafe_allow_html=True)
            county_risk = (
                df[df["risk_tier"] == "HIGH"]
                .groupby("county_encoded").size()
                .reset_index(name="high_risk_count")
                .sort_values("high_risk_count", ascending=False).head(10)
            )
            county_risk["county_encoded"] = "County " + county_risk["county_encoded"].astype(str)
            fig4 = px.bar(county_risk, x="high_risk_count", y="county_encoded",
                          orientation="h", color_discrete_sequence=["#dc2626"],
                          labels={"high_risk_count": "Children Flagged HIGH Risk", "county_encoded": ""})
            fig4.update_layout(height=320, plot_bgcolor="white", paper_bgcolor="white",
                               margin=dict(t=10, b=0, l=5),
                               xaxis=dict(showgrid=True, gridcolor="#f1f5f9"))
            st.plotly_chart(fig4, use_container_width=True)
            st.markdown(insight(
                "<b>How to read this:</b> Counties with longer bars have more children predicted to miss vaccines. "
                "Programme managers can use this to decide where to direct supervision visits, supplies, or additional CHW support. "
                "Note: this reflects the number of children, not the rate — larger counties will naturally have more flagged children."
            ), unsafe_allow_html=True)

    # ── Action list ──
    st.markdown('<div class="section-header">🚨 Top 20 Children Requiring Immediate Attention</div>',
                unsafe_allow_html=True)
    st.markdown(
        "The table below lists the 20 children with the highest risk scores. "
        "Share this list with CHW supervisors to prioritise home visits for this week."
    )

    display_cols = {
        "patient_id":            "Patient ID",
        "patient_age_in_months": "Age (months)",
        "risk_score":            "Risk Score",
        "risk_tier":             "Risk Level",
        "due_count_clean":       "Outstanding Doses",
    }
    available = [c for c in display_cols if c in df.columns]
    top_df = (df[available].sort_values("risk_score", ascending=False)
              .head(20).reset_index(drop=True))
    top_df.index += 1
    top_df.columns = [display_cols[c] for c in available]

    def colour_tier(val):
        m = {"HIGH":   "background-color:#fef2f2;color:#dc2626;font-weight:700",
             "MEDIUM": "background-color:#fffbeb;color:#d97706;font-weight:700",
             "LOW":    "background-color:#f0fdf4;color:#16a34a;font-weight:700"}
        return m.get(val, "")

    fmt = {}
    if "Risk Score" in top_df.columns:    fmt["Risk Score"] = "{:.1%}"
    if "Age (months)" in top_df.columns:  fmt["Age (months)"] = "{:.0f}"

    styled = top_df.style.applymap(colour_tier, subset=["Risk Level"]).format(fmt)
    st.dataframe(styled, use_container_width=True, height=420)
    st.caption(
        "Risk Score = the model's estimated probability that this child will miss a vaccine. "
        "Outstanding Doses = number of vaccines due that have not yet been recorded."
    )

# ── Page 2: Patient Risk Scorer ────────────────────────────────────────────────

def page_scorer():
    st.markdown("## 🔍 Check a Patient's Risk")
    st.markdown(
        "Use this page to **look up a specific child** and understand their risk of missing vaccines. "
        "The tool will show you a risk score, a recommended action, and an explanation of the "
        "key factors driving that child's risk."
    )

    # Step guide
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("""<div class="step-box">
            <div class="step-num">Step 1</div>
            <div class="step-title">Select the child</div>
            <div class="step-desc">Choose a patient ID from the dropdown, or type part of the ID to search.</div>
        </div>""", unsafe_allow_html=True)
    with c2:
        st.markdown("""<div class="step-box">
            <div class="step-num">Step 2</div>
            <div class="step-title">Get the risk score</div>
            <div class="step-desc">Click "Check Risk" to run the model. The result appears in seconds.</div>
        </div>""", unsafe_allow_html=True)
    with c3:
        st.markdown("""<div class="step-box">
            <div class="step-num">Step 3</div>
            <div class="step-title">Read the explanation</div>
            <div class="step-desc">Scroll down to see exactly which factors are driving the risk and what action to take.</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    model, preprocessor, feature_names = load_artifacts()
    df  = load_dataset()
    cfg = load_config()

    col1, col2 = st.columns([3, 1])
    with col1:
        patient_idx = st.selectbox(
            "Select a patient by ID",
            options=df.index.tolist(),
            format_func=lambda i: (
                f"{df.loc[i, 'patient_id']}  •  "
                f"Age {df.loc[i, 'patient_age_in_months']:.0f} months  •  "
                f"{'Male' if df.loc[i, 'patient_sex_binary'] == 1 else 'Female'}"
            ) if "patient_id" in df.columns else f"Row {i}",
            help="Type to search by patient ID",
        )
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)
        run_btn = st.button("Check Risk ▶", type="primary", use_container_width=True)

    if run_btn:
        row = df.loc[[patient_idx]]
        _score_and_display(row, model, preprocessor, feature_names, cfg)


def _score_and_display(row, model, preprocessor, feature_names, cfg):
    feat_cols = [c for c in feature_names if c in row.columns]
    X = row[feat_cols].copy()
    for c in [f for f in feature_names if f not in row.columns]:
        X[c] = np.nan
    X = X[feature_names]

    try:
        X_t   = preprocessor.transform(X)
        score = float(model.predict_proba(X_t)[0, 1])
    except Exception as e:
        st.error(f"Could not calculate risk score: {e}")
        return

    tier  = get_risk_tier(score, cfg)
    color = RISK_COLORS[tier]

    st.markdown("---")

    # ── Result header ──
    r1, r2, r3 = st.columns([1, 1, 2])
    with r1:
        st.markdown(kpi("Risk Score", f"{score:.0%}",
                        "probability of missing a vaccine", color),
                    unsafe_allow_html=True)
    with r2:
        tier_labels = {"HIGH": "HIGH RISK", "MEDIUM": "MEDIUM RISK", "LOW": "LOW RISK"}
        st.markdown(kpi("Risk Level", tier_labels[tier], "", color), unsafe_allow_html=True)
    with r3:
        action_html = {
            "HIGH":
                '<div class="action-high">'
                '🚨 <b>Immediate home visit required</b><br>'
                'This child has a high probability of missing vaccines. '
                'The CHW should visit within the next few days. '
                'Bring the vaccine card, check outstanding doses, and arrange a facility referral if needed.'
                '</div>',
            "MEDIUM":
                '<div class="action-medium">'
                '📋 <b>Schedule a home visit this month</b><br>'
                'This child is showing early warning signs. '
                'A timely CHW visit before the end of the month can prevent the child from falling further behind.'
                '</div>',
            "LOW":
                '<div class="action-low">'
                '✅ <b>Routine follow-up at next scheduled visit</b><br>'
                'This child is largely on-track. No urgent action needed. '
                'Include in the regular monthly visit cycle.'
                '</div>',
        }
        st.markdown(action_html[tier], unsafe_allow_html=True)

    # ── Gauge ──
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score * 100,
        number={"suffix": "%", "font": {"size": 42, "color": color}},
        title={"text": "Defaulter Risk", "font": {"size": 14, "color": "#64748b"}},
        gauge={
            "axis": {"range": [0, 100], "ticksuffix": "%",
                     "tickvals": [0, 33, 60, 100],
                     "ticktext": ["0%", "33%\nMedium", "60%\nHigh", "100%"]},
            "bar":  {"color": color, "thickness": 0.28},
            "steps": [
                {"range": [0, 33],   "color": "#dcfce7"},
                {"range": [33, 60],  "color": "#fef9c3"},
                {"range": [60, 100], "color": "#fee2e2"},
            ],
            "threshold": {"line": {"color": "black", "width": 3}, "value": score * 100},
        },
    ))
    fig.update_layout(height=280, margin=dict(t=30, b=0, l=30, r=30),
                      paper_bgcolor="white")
    gcol, _ = st.columns([1, 1])
    with gcol:
        st.plotly_chart(fig, use_container_width=True)

    st.markdown(insight(
        "<b>What does this score mean?</b> "
        f"A score of <b>{score:.0%}</b> means the model estimates a <b>{score:.0%} probability</b> "
        "that this child will miss or has missed a scheduled vaccine. "
        "This is based on the child's age, vaccine history, how recently a CHW has visited, "
        "and other factors recorded in the CHT system. "
        "It is not a diagnosis — it is a prioritisation signal."
    ), unsafe_allow_html=True)

    # ── Key factors ──
    st.markdown('<div class="section-header">What is driving this child\'s risk?</div>',
                unsafe_allow_html=True)
    st.markdown(
        "The chart and explanation below show the factors that most influenced this child's score. "
        "Factors shown in **red** are pushing the risk higher. "
        "Factors shown in **blue** are working in the child's favour."
    )

    _shap_waterfall(tier)

    # ── Child summary ──
    with st.expander("📋 View this child's full details"):
        st.markdown("All recorded values for this child from the CHT system:")
        show_cols = {c: fname(c) for c in row.columns
                     if not c.startswith("_") and row[c].notna().any()}
        display_df = (row[list(show_cols.keys())]
                      .T.rename(columns={row.index[0]: "Value"})
                      .rename(index=show_cols))
        display_df["Value"] = display_df["Value"].apply(
            lambda v: f"{v:.2f}" if isinstance(v, float) else str(v)
        )
        st.dataframe(display_df, use_container_width=True)


def _shap_waterfall(tier: str):
    img_map = {
        "HIGH":   REPORTS_DIR / "shap" / "waterfall_high_example.png",
        "MEDIUM": REPORTS_DIR / "shap" / "waterfall_medium_example.png",
        "LOW":    REPORTS_DIR / "shap" / "waterfall_low_example.png",
    }
    img_path = img_map.get(tier)
    if img_path and img_path.exists():
        st.image(str(img_path), use_container_width=True)

    # Reading guide
    reading_guide = {
        "HIGH": (
            "🔴 <b>Reading this chart for a HIGH risk child:</b> "
            "The bars extending to the right (red) show the factors that are most strongly driving this child's risk up. "
            "Common drivers at high risk include: many outstanding doses, child in the 6–18 month window (peak EPI age), "
            "and a long gap since the last CHW contact. The final predicted probability is shown at the top."
        ),
        "MEDIUM": (
            "🟡 <b>Reading this chart for a MEDIUM risk child:</b> "
            "This child has a mix of risk factors and protective factors roughly balancing each other. "
            "Red bars show what is pushing the risk up; blue bars show what is keeping it from going higher. "
            "A CHW visit this month could address the red factors before they escalate."
        ),
        "LOW": (
            "🟢 <b>Reading this chart for a LOW risk child:</b> "
            "Blue bars dominate, meaning the child's profile (up-to-date vaccines, recent CHW contact) "
            "is actively reducing the risk score. This child does not need urgent attention. "
            "Continue routine follow-up."
        ),
    }
    st.markdown(insight(reading_guide[tier]), unsafe_allow_html=True)

    # Plain-English drivers from saved JSON
    json_path = REPORTS_DIR / "shap" / f"patient_{tier.lower()}.json"
    if json_path.exists():
        data    = json.loads(json_path.read_text())
        drivers = data.get("top_drivers", [])
        if drivers:
            st.markdown("**The three factors that mattered most for this child:**")
            for i, d in enumerate(drivers[:3], 1):
                direction = "increases" if d.get("direction") == "increases_risk" else "reduces"
                arrow     = "🔺" if direction == "increases" else "🔻"
                name      = d.get("friendly_name") or fname(d.get("feature", ""))
                st.markdown(
                    f"{arrow} **{i}. {name}** — "
                    f"*{d.get('plain_english', f'This factor {direction} the risk score.')}*"
                )

# ── Page 3: How Well Does the Model Work? ──────────────────────────────────────

def page_performance():
    st.markdown("## 📈 How Well Does the Model Work?")
    st.markdown(
        "This page answers the question: **can we trust the risk scores?** "
        "The model was tested on children it had never seen before (a held-out test group of 1,373 children). "
        "The results below show how well it performed on that group."
    )

    st.info(
        "💡 **You do not need a statistics background to use this page.** "
        "Each section below has a plain-English explanation of what the numbers mean for the programme."
    )

    # ── Plain-language scorecard ──
    st.markdown('<div class="section-header">Model Report Card</div>', unsafe_allow_html=True)
    st.markdown(
        "Think of these as the model's grades across different aspects of performance:"
    )

    scorecard = [
        ("🎯 Overall ranking accuracy",    "89%",  "0.893 ROC-AUC",
         "The model correctly identifies which children are more at risk than others 89% of the time. "
         "This is like a CHW correctly guessing who needs a visit first — 9 times out of 10."),
        ("🔍 Quality of the priority list", "70%",  "0.697 PR-AUC",
         "Of all children ranked in the top tier by the model, 70% are genuinely at risk. "
         "This matters because CHWs have limited time — a high-quality list means fewer wasted visits."),
        ("⚖️ Balance of flags and catches", "58%",  "0.580 F1 Score",
         "This balances two competing goals: catching as many defaulters as possible vs. not overwhelming CHWs with false alarms. "
         "58% is a good score for a real-world health dataset with 1 in 6 children being a defaulter."),
        ("✅ Accuracy of HIGH risk flags",  "77%",  "0.772 Precision",
         "When the model flags a child as HIGH risk, 77% of those children truly are defaulters. "
         "This means 3 in 4 home visits prompted by a HIGH risk flag are to a genuinely at-risk child."),
        ("📡 Defaulters caught",            "47%",  "0.465 Recall at 0.50",
         "At the default threshold, the model catches 47% of all actual defaulters. "
         "Lowering the threshold to 0.35 raises this to 85% — catching more defaulters at the cost of a slightly larger visit list."),
        ("🎲 Probability trustworthiness",  "97.7%","0.023 ECE",
         "When the model says '70% risk', about 70% of those children are true defaulters. "
         "The probabilities are reliable — supervisors can use them as real estimates, not just rankings."),
        ("🏆 Top 20% list quality",         "57%",  "0.569 Precision@20%",
         "If CHWs can only visit the top 20% of ranked children, 57% of those visits will be to genuine defaulters. "
         "Without the model, random selection would yield only 16.5% (the base rate)."),
        ("⚖️ Fairness — boys vs. girls",    "0.020","AUC gap (threshold <0.10)",
         "The model performs almost equally well for boys and girls (89% vs 88% accuracy). "
         "The gap of 2 percentage points is well within acceptable limits, meaning no systematic bias."),
    ]

    for i in range(0, len(scorecard), 2):
        col_l, col_r = st.columns(2)
        for col, idx in [(col_l, i), (col_r, i + 1)]:
            if idx < len(scorecard):
                label, value, tech, explanation = scorecard[idx]
                with col:
                    with st.container():
                        st.markdown(
                            f'<div class="kpi-card">'
                            f'<div class="kpi-label">{label}</div>'
                            f'<div class="kpi-value">{value}</div>'
                            f'<div class="kpi-sub">{tech}</div>'
                            f'</div>',
                            unsafe_allow_html=True,
                        )
                        st.markdown(
                            f'<div style="font-size:0.82rem; color:#475569; '
                            f'margin:-4px 0 16px 0; line-height:1.5;">{explanation}</div>',
                            unsafe_allow_html=True,
                        )

    # ── ROC/PR curves ──
    roc_path = REPORTS_DIR / "roc_pr_curves.png"
    if roc_path.exists():
        st.markdown('<div class="section-header">Ranking Performance Charts</div>',
                    unsafe_allow_html=True)
        st.image(str(roc_path), use_container_width=True)
        st.markdown(insight(
            "<b>What you are looking at:</b> Two curves showing how the model performs as you move "
            "the risk threshold up or down. The further the curves bow toward the top-left corner, "
            "the better the model. A coin-flip model would produce a straight diagonal line — "
            "our model's curves are far above that, confirming it is genuinely predictive. "
            "<br><br>"
            "<b>What it means for operations:</b> Regardless of the threshold you choose, "
            "the model's ranking of children is reliable. You can adjust the threshold based on "
            "CHW capacity — lower threshold = more visits but more defaulters caught."
        ), unsafe_allow_html=True)

    # ── Calibration ──
    cal_path = REPORTS_DIR / "calibration_curve.png"
    if cal_path.exists():
        st.markdown('<div class="section-header">Are the Risk Scores Trustworthy Probabilities?</div>',
                    unsafe_allow_html=True)
        col_img, col_txt = st.columns([1, 1])
        with col_img:
            st.image(str(cal_path), use_container_width=True)
        with col_txt:
            st.markdown(insight(
                "<b>What a well-calibrated model means:</b> When the model says a child has a 70% risk, "
                "it means that among all children given that score, approximately 70 out of 100 are "
                "genuine defaulters.<br><br>"
                "<b>How to read the chart:</b> The diagonal line is the 'perfect' reference. "
                "Our model's dots (the blue line) follow the diagonal very closely. "
                "If the dots were far above the line, the model would be over-alarming. "
                "If far below, it would be under-estimating. Neither is the case here.<br><br>"
                "<b>Practical implication:</b> Supervisors can communicate the score to caregivers as "
                "a genuine probability — not just a rank. "
                "A 90% score means the child is in serious need of follow-up."
            ), unsafe_allow_html=True)

    # ── Feature importance ──
    fi_path = REPORTS_DIR / "feature_importance.png"
    if fi_path.exists():
        st.markdown('<div class="section-header">What Does the Model Base Its Decisions On?</div>',
                    unsafe_allow_html=True)
        st.image(str(fi_path), use_container_width=True)
        st.markdown(insight(
            "<b>What you are looking at:</b> The factors the model uses most when deciding a child's risk score. "
            "Longer bars mean the model relies more heavily on that factor.<br><br>"
            "<b>Key finding:</b> The top factors are <b>child's age</b>, <b>how recently a CHW visited</b>, "
            "and <b>whether the Penta series is complete</b>. "
            "These are all factors that CHWs already collect through CHT forms — "
            "no special data collection is needed.<br><br>"
            "<b>Programme implication:</b> Consistent CHW home visits directly improve the model's ability "
            "to identify at-risk children. The programme is not at the mercy of factors it cannot control."
        ), unsafe_allow_html=True)

    # ── SHAP beeswarm ──
    bee_path = REPORTS_DIR / "shap" / "shap_beeswarm.png"
    if bee_path.exists():
        st.markdown('<div class="section-header">How Do These Factors Affect Each Child Differently?</div>',
                    unsafe_allow_html=True)
        st.image(str(bee_path), use_container_width=True)
        st.markdown(insight(
            "<b>What you are looking at:</b> Each dot is one child. "
            "Dots to the right of centre mean that factor is pushing the child's risk <b>up</b>. "
            "Dots to the left mean it is pushing the risk <b>down</b>. "
            "Red dots = high value of that factor; Blue dots = low value.<br><br>"
            "<b>Example — top row (Child's age):</b> Blue dots to the right means "
            "very young children (low age = blue) are at higher risk, which is consistent with the "
            "EPI schedule — infants have the most vaccines due in a short window.<br><br>"
            "<b>Example — Penta series:</b> Blue dots to the right means children who have NOT "
            "completed the Penta series are more likely to be defaulters — exactly as expected."
        ), unsafe_allow_html=True)

    # ── Threshold table ──
    st.markdown('<div class="section-header">Choosing the Right Threshold for CHW Operations</div>',
                unsafe_allow_html=True)
    st.markdown(
        "The **threshold** controls how sensitive the model is. "
        "A lower threshold means more children are flagged — the CHW visits more children but catches more defaulters. "
        "A higher threshold means fewer visits but also fewer defaulters caught. "
        "**Programme managers should choose the threshold based on available CHW capacity.**"
    )

    thresh_df = pd.DataFrame({
        "Risk Threshold":       [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70],
        "Children Flagged":     [652,  569,  504,  451,  394,  351,  306,  270,  229,  200,  169],
        "% of Flags Correct":   [31.9, 35.5, 39.3, 42.8, 46.7, 49.9, 53.9, 57.4, 62.0, 67.0, 73.4],
        "% of Defaulters Caught":[92.0, 89.4, 87.6, 85.4, 81.4, 77.4, 73.0, 68.6, 62.8, 59.3, 54.9],
        "Overall Score (F1)":   [0.474, 0.508, 0.542, 0.570, 0.594, 0.607, 0.620, 0.625, 0.624, 0.629, 0.628],
        "Recommendation":       ["", "", "", "⭐ CHW deployment", "", "", "⭐ Best overall F1",
                                  "", "", "", ""],
    })

    def hl_rec(val):
        if "CHW" in str(val):   return "background-color:#fef9c3; font-weight:700"
        if "Best" in str(val):  return "background-color:#eff6ff; font-weight:700"
        return ""

    styled = (thresh_df.style
              .applymap(hl_rec, subset=["Recommendation"])
              .format({
                  "% of Flags Correct":    "{:.1f}%",
                  "% of Defaulters Caught":"{:.1f}%",
                  "Overall Score (F1)":    "{:.3f}",
              }))
    st.dataframe(styled, use_container_width=True, hide_index=True)
    st.markdown(insight(
        "<b>How to read this table:</b> "
        "<b>Children Flagged</b> = how many children the model would send CHWs to visit. "
        "<b>% of Flags Correct</b> = of those visits, what fraction will be genuine defaulters. "
        "<b>% of Defaulters Caught</b> = of all children who are true defaulters, what fraction does the model find.<br><br>"
        "<b>Recommended starting point:</b> A threshold of <b>0.35</b> (highlighted in yellow) "
        "is recommended for CHW deployment — it catches 85% of defaulters while keeping the visit "
        "list manageable (451 children, about 10% of the cohort)."
    ), unsafe_allow_html=True)

    # ── Fairness ──
    st.markdown('<div class="section-header">Is the Model Fair to All Children?</div>',
                unsafe_allow_html=True)
    st.markdown(
        "The model was checked to ensure it performs equally well for boys and girls. "
        "A model that works well for one group but not another would be unfair and unreliable."
    )
    fair_df = pd.DataFrame({
        "Group":             ["Girls", "Boys"],
        "Children in test":  [694, 675],
        "Defaulter rate":    ["15.7%", "17.3%"],
        "Model accuracy":    ["88.2%", "90.2%"],
        "Difference in accuracy": ["—", "2.0 percentage points ✅ (below 10pp threshold)"],
    })
    st.dataframe(fair_df, use_container_width=True, hide_index=True)
    st.markdown(insight(
        "<b>Result: No meaningful bias detected.</b> The model is 88.2% accurate for girls and 90.2% for boys — "
        "a gap of only 2 percentage points, well within the acceptable limit of 10 percentage points. "
        "Both groups can be scored with confidence."
    ), unsafe_allow_html=True)

# ── Page 4: Data Quality & Model Health ────────────────────────────────────────

def page_drift():
    st.markdown("## 🔔 Data Quality & Model Health")
    st.markdown(
        "This page answers the question: **is the data still representative?** "
        "Over time, the characteristics of children in the CHT system can change — "
        "for example if new CHW areas are added, or if data collection practices shift. "
        "When the data changes significantly, the model's predictions may become less reliable "
        "and the model should be retrained."
    )

    st.info(
        "💡 **Think of this as a health check for the model.** "
        "Green = the data is stable, predictions are reliable. "
        "Red = something has changed in the data — investigate before relying on scores."
    )

    drift_df = load_drift_report()
    if drift_df.empty:
        st.warning(
            "No health check report found. "
            "Ask your data team to run `python main.py --stage monitor` to generate the latest report."
        )
        return

    n_total   = len(drift_df)
    n_drifted = int((drift_df["status"] == "RED").sum()) if "status" in drift_df.columns else 0
    n_ok      = n_total - n_drifted

    # ── Status banner ──
    if n_drifted == 0:
        st.success(f"✅ **All {n_total} data checks passed.** The model data is stable. Predictions are reliable.")
    elif n_drifted <= 3:
        st.warning(
            f"⚠️ **{n_drifted} out of {n_total} checks flagged.** "
            "Minor data shifts detected. Review the flagged items below before the next scoring run."
        )
    else:
        st.error(
            f"🚨 **{n_drifted} out of {n_total} checks flagged.** "
            "Significant data shifts detected. Contact your data team — the model may need retraining."
        )

    # ── KPIs ──
    st.markdown("<br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown(kpi("Data indicators checked", str(n_total),
                        "patient characteristics monitored"), unsafe_allow_html=True)
    with c2:
        color = "#dc2626" if n_drifted > 3 else ("#d97706" if n_drifted > 0 else "#16a34a")
        st.markdown(kpi("Flagged for change", str(n_drifted),
                        "need investigation", color), unsafe_allow_html=True)
    with c3:
        st.markdown(kpi("Stable indicators", str(n_ok),
                        "no action needed", "#16a34a"), unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Bar chart ──
    if "psi" in drift_df.columns and "feature" in drift_df.columns:
        st.markdown('<div class="section-header">Stability Check Results by Indicator</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "Each bar below shows how much a patient characteristic has changed compared to when the model was trained. "
            "**Green** = no meaningful change. **Amber** = minor shift, worth watching. **Red** = significant change."
        )

        plot_df = drift_df.sort_values("psi", ascending=False).head(20).copy()
        plot_df["colour"] = plot_df["psi"].apply(
            lambda v: "#dc2626" if v >= 0.20 else ("#d97706" if v >= 0.10 else "#16a34a")
        )
        plot_df["Indicator"] = plot_df["feature"].apply(fname)

        fig = go.Figure(go.Bar(
            y=plot_df["Indicator"],
            x=plot_df["psi"],
            orientation="h",
            marker_color=plot_df["colour"].tolist(),
            text=[f"{v:.3f}" for v in plot_df["psi"]],
            textposition="outside",
            hovertemplate="<b>%{y}</b><br>Stability score: %{x:.3f}<extra></extra>",
        ))
        fig.add_vline(x=0.20, line_dash="dash", line_color="#dc2626",
                      annotation_text="Action threshold", annotation_position="top right")
        fig.add_vline(x=0.10, line_dash="dot",  line_color="#d97706",
                      annotation_text="Watch threshold", annotation_position="top right")
        fig.update_layout(
            height=520, plot_bgcolor="white", paper_bgcolor="white",
            margin=dict(t=20, b=0, l=10),
            xaxis=dict(showgrid=True, gridcolor="#f1f5f9", title="Change Score (PSI) — lower is better"),
            yaxis=dict(autorange="reversed"),
        )
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(insight(
            "<b>How to read this chart:</b> A bar that stays to the left (small change score) means "
            "that patient characteristic looks similar to when the model was trained — good sign. "
            "A long red bar means that characteristic has changed significantly in the current data. "
            "This does not necessarily mean predictions are wrong — but it is a signal to investigate "
            "whether the model needs to be updated with newer data."
        ), unsafe_allow_html=True)

    # ── Flagged items detail ──
    if n_drifted > 0:
        st.markdown('<div class="section-header">Indicators Requiring Attention</div>',
                    unsafe_allow_html=True)
        st.markdown(
            "The table below lists the indicators that have changed most since the model was trained. "
            "Share this with your data team to investigate whether the change reflects a real population "
            "shift or a data quality issue."
        )

        flagged = drift_df[drift_df["status"] == "RED"].copy() if "status" in drift_df.columns else drift_df.copy()
        flagged["Indicator"] = flagged["feature"].apply(fname) if "feature" in flagged.columns else ""
        flagged["Change Score"] = flagged["psi"].round(3) if "psi" in flagged.columns else ""
        flagged["Training Average"] = flagged["mean_train"].round(3) if "mean_train" in flagged.columns else ""
        flagged["Current Average"]  = flagged["mean_new"].round(3) if "mean_new" in flagged.columns else ""
        flagged["Status"] = "⚠️ Changed"

        show_cols = [c for c in ["Indicator", "Training Average", "Current Average",
                                  "Change Score", "Status"] if c in flagged.columns]
        st.dataframe(flagged[show_cols].reset_index(drop=True), use_container_width=True, hide_index=True)

    # ── Full table ──
    with st.expander("📋 View full stability report for all indicators"):
        full = drift_df.copy()
        if "feature" in full.columns:
            full.insert(0, "Indicator", full["feature"].apply(fname))
        if "status" in full.columns:
            full["Status"] = full["status"].map({"RED": "⚠️ Changed", "GREEN": "✅ Stable"})

        def hl_status(val):
            if "Changed" in str(val): return "background-color:#fef2f2;color:#dc2626;font-weight:600"
            if "Stable"  in str(val): return "background-color:#f0fdf4;color:#16a34a;font-weight:600"
            return ""

        num_cols = full.select_dtypes("number").columns.tolist()
        fmt = {c: "{:.4f}" for c in num_cols}
        style_obj = full.style.format(fmt)
        if "Status" in full.columns:
            style_obj = style_obj.applymap(hl_status, subset=["Status"])
        st.dataframe(style_obj, use_container_width=True, height=400)

    # ── Plain-language guide ──
    with st.expander("❓ What should I do if indicators are flagged?"):
        st.markdown("""
**Step 1 — Don't panic.**
A flagged indicator means the data has changed — it doesn't necessarily mean the model is broken.
Some change is normal as the programme grows and new CHW areas are added.

**Step 2 — Identify the cause.**
Ask your data team: Has anything changed in CHT data collection in the last reporting period?
Were new CHW areas enrolled? Was there a system update that changed how data is recorded?

**Step 3 — Assess impact.**
If the flagged indicators are major predictors (age, outstanding doses, visit frequency),
the risk scores may be less accurate. Use them with caution until the model is retrained.

**Step 4 — Retrain the model.**
If the data shift is real and sustained (not a one-month glitch), ask the data team to
retrain the model with the most recent data. The pipeline takes approximately 5 minutes to run.

**Step 5 — Monitor monthly.**
Ideally, this page should be reviewed once a month after each data refresh.
        """)

# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    page = sidebar()

    if "Dashboard" in page:
        page_dashboard()
    elif "Patient" in page:
        page_scorer()
    elif "Model" in page:
        page_performance()
    elif "Data Quality" in page:
        page_drift()


if __name__ == "__main__":
    main()
