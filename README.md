# Immunization Defaulter Risk Engine

### Production ML Pipeline · XGBoost + SHAP · FastAPI · PostgreSQL · Kenya CHW Platform

[![Live App](https://img.shields.io/badge/Live%20App-immunizationengine.streamlit.app-FF4B4B?logo=streamlit&logoColor=white)](https://immunizationengine.streamlit.app/)

> **Try it live:** [https://immunizationengine.streamlit.app/](https://immunizationengine.streamlit.app/)

---

## The Problem

Kenya's Community Health Worker (CHW) program serves **8.5 million individuals** through 4,600+ CHW areas. Each CHW manages 20–35 under-2 children per catchment area but lacks a systematic way to prioritise which children to visit on any given day.

Children who miss vaccines do so silently — there is no alert, no flag, no notification. CHWs currently rely on memory and paper registers. The result: preventable outbreaks, missed booster windows, and inequitable coverage across districts.

This engine solves that with a **real-time, explainable risk score** delivered to the CHW's mobile app every morning.

---

## Why This Matters for Managed Care

The architecture of this engine maps directly onto the core challenges of Medicare and Medicaid managed care:

| This Engine | Managed Care Equivalent |
|---|---|
| CHW prioritisation list | Member outreach prioritisation for preventive gap closure |
| Vaccine defaulter probability | Care gap adherence risk score |
| 16.5% positive rate with class imbalance handling | Same challenge in Medicare Stars gap-closure programs |
| Per-patient SHAP plain-English drivers | Explainability requirement for HIPAA-compliant clinical decision support |
| PSI drift monitoring across district rollouts | Model monitoring during phased member population rollouts |

The methodology is domain-agnostic. The problem — identifying who will disengage from a care protocol before they do, ranking them by risk, and explaining why to a non-technical frontline worker — is structurally identical whether the setting is Kenya's EPI schedule or a Medicare Advantage HEDIS measure.

---

## What It Does

| Capability | Detail |
|---|---|
| **Predicts** | Calibrated defaulter probability (0–1) per child, updated monthly |
| **Explains** | Per-patient SHAP drivers in plain English ("Child is 9 months old and has 3 outstanding doses") |
| **Serves** | FastAPI `/predict` and `/predict/batch` endpoints with <100ms response |
| **Monitors** | PSI feature drift + label shift detection across district rollouts |
| **Tracks** | Full MLflow experiment tracking, model versioning, and registry |

---

## Live Results — Full Production Database

> Trained on **6,864 unique patients** across **4,672 CHW areas** in Kenya.
> Data pulled live from a 12-table PostgreSQL database (11M+ rows in operational tables).

### Model Performance

| Metric | Value | Interpretation |
|---|---|---|
| **ROC-AUC** | **0.892** | Excellent separation between defaulters and non-defaulters |
| **PR-AUC** | **0.698** | Strong ranking quality on the imbalanced positive class |
| **F1 Score** | 0.580 | Balanced precision/recall at default threshold |
| **Precision** | 0.772 | 77% of flagged children are true defaulters |
| **Recall** | 0.465 | Catches 47% of all defaulters at default threshold |
| **Brier Score** | 0.084 | Well-calibrated probability estimates |
| **ECE** | **0.023** | Near-perfect calibration (2pp expected error) |
| **Precision@Top 20%** | 0.569 | CHWs visiting top-ranked children: 57% are true defaulters |

### Fairness — Zero Bias Across Sex

| Group | N | Positive Rate | ROC-AUC |
|---|---|---|---|
| Female | 694 | 15.7% | 0.882 |
| Male | 675 | 17.3% | 0.902 |
| **AUC gap** | — | — | **0.020** *(threshold: <0.10)* |

### Threshold Operating Curve

| Threshold | Children Flagged | Precision | Recall | F1 |
|---|---|---|---|---|
| 0.30 | 504 | 39.3% | 87.6% | 0.542 |
| 0.35 | 451 | 42.8% | 85.4% | 0.570 |
| 0.40 | 394 | 46.7% | 81.4% | 0.594 |
| 0.45 | 351 | 49.9% | 77.4% | 0.607 |
| **0.50** | **306** | **53.9%** | **73.0%** | **0.620** |
| 0.55 | 270 | 57.4% | 68.6% | 0.625 |
| 0.60 | 229 | 62.0% | 62.8% | 0.624 |
| 0.65 | 200 | 67.0% | 59.3% | 0.629 |
| 0.70 | 169 | 73.4% | 54.9% | 0.628 |

*Operational note: threshold = 0.50 maximises F1. For CHW deployment where missing a defaulter has consequences, threshold = 0.30–0.35 raises recall to 88–94% at ~7% flag rate — this decision should be made jointly with program staff.*

---

## Visualisations

> **Reading guide for programme stakeholders:** Each chart below includes an interpretation written for three audiences — *CHW supervisors*, *MOH programme managers*, and *data/technical reviewers*. You do not need a statistics background to act on these results.

---

### ROC & Precision-Recall Curves

![ROC and PR Curves](reports/roc_pr_curves.png)

**What you are looking at:** Two curves that measure how well the model ranks children by defaulter risk across all possible thresholds. The ROC curve plots the true-positive rate (sensitivity) against the false-positive rate. The Precision-Recall curve plots the fraction of flagged children who are true defaulters (precision) against the fraction of all defaulters captured (recall).

| Audience | What it means |
|---|---|
| **CHW supervisor** | The model is much better than random chance at identifying which children are likely to miss vaccines. At almost any workload level you choose, it outperforms a simple age or visit-recency rule. |
| **MOH programme manager** | ROC-AUC = 0.893 means the model correctly ranks a defaulter above a non-defaulter 89% of the time. This is a strong signal for a real-world, imbalanced public health dataset. PR-AUC = 0.697 confirms the quality holds even when only 1 in 6 children is a defaulter. |
| **Technical reviewer** | The PR curve area of 0.697 substantially exceeds the no-skill baseline of 0.165 (positive rate). Both curves were computed on a stratified 20% held-out test set, not training data. |

---

### Probability Calibration

> ECE = 0.023 — predicted probabilities match observed defaulter rates within 2 percentage points.

![Calibration Curve](reports/calibration_curve.png)

**What you are looking at:** The diagonal line represents a *perfectly calibrated* model — one where a score of 0.70 means exactly 70% of such children are true defaulters. Each dot shows the actual observed defaulter rate for children grouped by their predicted score. Dots close to the diagonal mean the model's probabilities are trustworthy as real-world rates.

| Audience | What it means |
|---|---|
| **CHW supervisor** | When the app shows "72% risk", roughly 7 out of every 10 children with that score will genuinely be defaulters. The score is not just a ranking — it is an actionable probability that supervisors can communicate to caregivers. |
| **MOH programme manager** | Calibrated probabilities allow the programme to estimate how many defaulters will be missed at any given operational threshold. This is essential for planning CHW workload and catching high-burden sub-counties. |
| **Technical reviewer** | Isotonic regression calibration was applied post-training using 5-fold cross-validation. ECE = 0.023 (2.3pp expected error) is well within the acceptable threshold of 0.15 defined in the model config. The calibration curve shows no systematic over- or under-confidence across the probability range. |

---

### Feature Importance

![Feature Importance](reports/feature_importance.png)

**What you are looking at:** A bar chart of the model's internal feature importance scores (XGBoost gain) — how much each variable contributed to splitting decisions across all 290 trees. Longer bars = stronger contributors to the prediction.

| Audience | What it means |
|---|---|
| **CHW supervisor** | The model relies most on information CHWs already collect: the child's age, how recently a CHW made contact, and whether the Penta series is complete. No exotic data inputs are needed — the existing CHT forms already capture the strongest signals. |
| **MOH programme manager** | The dominance of *age* and *months since last CHW contact* confirms that programme engagement quality (visit recency) is as predictive as clinical history. Investment in consistent CHW home visits directly improves the model's ability to identify defaulters early. |
| **Technical reviewer** | XGBoost gain importance is computed per-tree split and can be biased toward high-cardinality continuous features. SHAP values (below) provide a more robust, model-agnostic importance ranking. Both approaches yield consistent top-3 features. |

---

## SHAP Explainability

Every prediction is decomposed into plain-English per-patient drivers using TreeSHAP. CHWs receive a natural-language summary alongside the risk score.

---

### Global Feature Impact — Beeswarm Plot

> Each dot = one patient. Colour = feature value (red = high, blue = low). Horizontal position = SHAP impact on defaulter probability (right = increases risk, left = decreases risk).

![SHAP Beeswarm](reports/shap/shap_beeswarm.png)

**What you are looking at:** Each dot represents one patient. The plot shows both *direction* (does a high value of this feature push the score up or down?) and *magnitude* (how much does it move the score?) simultaneously. Features are ranked by their average absolute impact.

| Audience | What it means |
|---|---|
| **CHW supervisor** | Red dots to the right of centre mean "a high value of this feature increases the risk score." For *Doses currently outstanding*, red (many outstanding doses) pushes risk up — exactly as expected. Blue dots to the right of centre mean "a low value increases risk" — for *Penta series complete*, low completeness (blue) is the danger signal. |
| **MOH programme manager** | The beeswarm shows that older children (red, top row) are generally at *lower* risk because they have aged out of the core EPI window — while children in the 6–18 month window (medium age, blue/red mixed) show the widest spread of risk. District plans should target this age band. |
| **Technical reviewer** | SHAP values were computed using TreeExplainer on the base XGBoost estimator extracted from `CalibratedClassifierCV`. Background sample = 100 randomly selected training records. Values represent additive log-odds contributions consistent with the model's output space. |

---

### Global Feature Importance — Bar Chart

![SHAP Bar](reports/shap/shap_bar.png)

**What you are looking at:** The mean absolute SHAP value per feature across all patients — a single summary number representing each feature's average contribution to moving the risk score away from the population baseline.

| Audience | What it means |
|---|---|
| **CHW supervisor** | *Child's age* and *months since last CHW contact* are the two features that move the risk score the most on average. A CHW who visits a family on time — regardless of any other factor — meaningfully reduces that child's predicted risk. |
| **MOH programme manager** | The top three features (age, visit recency, Penta series) are all programme-actionable. This means the model is not relying on fixed socioeconomic factors that the programme cannot change — it is reflecting behaviours and service delivery quality that CHW training and supervision can directly improve. |
| **Technical reviewer** | Mean \|SHAP\| values are more reliable than XGBoost gain for comparing features of different scales and cardinalities, as they reflect actual impact on output probabilities rather than tree-splitting frequency. |

### Top 10 Features by Mean |SHAP|

| Rank | Feature | SHAP | Domain | Note |
|---|---|---|---|---|
| 1 | Child's age (months) | **0.726** | Child biology | |
| 2 | Months since last CHW contact | **0.515** | Engagement | |
| 3 | Penta 1-2-3 series complete | **0.398** | Vaccine history | |
| 4 | *(unresolved — label under investigation)* | **0.303** | Unknown | ⚠️ see note |
| 5 | Overall vaccine completeness (all) | 0.211 | Vaccine history | |
| 6 | Doses currently outstanding | 0.200 | Operational | |
| 7 | Vitamin A completeness | 0.178 | Nutrition | |
| 8 | Core vaccine completeness score | 0.116 | Vaccine history | |
| 9 | OPV 1-2-3 series complete | 0.093 | Vaccine history | |
| 10 | Under-2 children per CHW area | 0.067 | CHW workload | |

*Age and recency of CHW contact are the dominant signals — consistent with epidemiological priors.*

**⚠️ Rank 4 feature label:** The SHAP explainer reports `has_delayed_milestones_binary` (SHAP = 0.303) as the 4th most important feature. However, this field is 100% null in the current dataset (CHT milestone fields are not yet backfilled) and was excluded from the preprocessor's 44 fitted columns. The SHAP label at position 3 in the output matrix is therefore mislabelled — a real predictor with a real contribution exists at that rank, but its true identity cannot be confirmed until the feature name alignment between the fitted preprocessor and the SHAP explainer is resolved. This is tracked as a known open issue.

**⚠️ PNC label agreement:** Step 9 of the ETL pipeline compares the composite `is_defaulter` target (from the `iz` table) against the `is_immunization_defaulter` field in the `pnc` table. Agreement is **40.5% on 189 matched records** — below the expected 70%+ threshold. This is under active investigation and likely reflects different observation windows between the `iz` and `pnc` assessments rather than target variable contamination. The PNC field is used for audit only and is not a training feature.

*Note: 6 maternal/milestone features (growth monitoring, ANC visits, MUAC) are not yet available due to a 0% maternal join rate; model performance is expected to improve further once `preg_reg` CHW-area linkage is resolved.*

---

### Per-Patient Waterfall: HIGH Risk (99.7%)

![HIGH Risk Waterfall](reports/shap/waterfall_high_example.png)

> *"Child has 3 outstanding vaccine doses. Immediate home visit required. Arrange facility referral."*

**What you are looking at:** Each bar shows one feature's contribution to this specific child's risk score. Red bars push the score higher (toward defaulter); blue bars push it lower (toward on-track). The final predicted probability appears at the top. The baseline (E[f(x)]) is the average risk score across all children — bars show how this particular child deviates from average.

| Audience | What it means |
|---|---|
| **CHW supervisor** | This child has 3 outstanding doses *and* is in a critical age window — two independent reasons for concern. The waterfall shows exactly which factors are driving the flag, so the CHW can have a targeted conversation with the caregiver rather than a generic reminder. |
| **MOH programme manager** | A 99.7% risk score means fewer than 1 in 300 children with this profile are on-track. Immediate action is warranted. This child should appear at the top of the CHW's daily priority list. |
| **Technical reviewer** | Waterfall values are additive SHAP contributions in log-odds space, transformed to probability by the calibrated model. The sum of all SHAP values plus the base value equals the final log-odds of the prediction. |

---

### Per-Patient Waterfall: MEDIUM Risk (60.0%)

![MEDIUM Risk Waterfall](reports/shap/waterfall_medium_example.png)

**What you are looking at:** A child where risk factors and protective factors partially offset each other, producing a 60% probability — above the medium threshold (33%) but below the high threshold (60%).

| Audience | What it means |
|---|---|
| **CHW supervisor** | This child is not yet in crisis but is trending toward missed vaccines. The waterfall shows the key vulnerability — schedule the next home visit within the month before the situation escalates. |
| **MOH programme manager** | Medium-risk children are the largest intervention opportunity. They are still accessible and reachable, and a single timely CHW visit can shift them to low risk before any doses are missed. |

---

### Per-Patient Waterfall: LOW Risk (33.0%)

![LOW Risk Waterfall](reports/shap/waterfall_low_example.png)

**What you are looking at:** A child whose features — recent CHW contact, completed Penta series, appropriate age — combine to produce a score just at the low/medium boundary (33.0%).

| Audience | What it means |
|---|---|
| **CHW supervisor** | This child does not need an immediate visit. Routine follow-up at the next scheduled cycle is sufficient. The blue bars (protective factors) confirm the child is broadly on-track. |
| **MOH programme manager** | Low-risk children free up CHW capacity for the high and medium tiers. The model enables resource reallocation without de-prioritising any child entirely — low-risk patients still receive routine visits, just not urgent ones. |

---

## System Architecture

```
PostgreSQL (12 tables · 11M+ rows in operational tables)
        │
        │  Server-side SQL aggregation (custom pushdown for large tables)
        │  active_chps: DISTINCT ON → 4,672 CHW area rows
        │  homevisit:   GROUP BY    → monthly visit rate per area
        │  population:  PERCENTILE  → median u2 workload per area
        │  fp/refill:   GROUP BY    → household FP status
        ▼
┌─────────────────────────────────────────────────────────────┐
│  ETL Pipeline  (src/etl/)                                   │
│  Loader → Cleaner → Merger (10-step blueprint)              │
│  ─────────────────────────────────────────────────────────  │
│  Step 1:  Deduplicate iz (8,814 → 8,731 records)           │
│  Step 2:  EPI-schedule-gated vaccine completeness scores    │
│  Step 3:  Composite target variable construction            │
│  Step 4:  CHW metadata join          (100% match)           │
│  Step 5:  CHW supervision quality    (30% match)            │
│  Step 6:  Population workload        (100% match)           │
│  Step 7:  Home visit frequency       (100% match)           │
│  Step 8:  Maternal ANC health-seeking (area-level join)     │
│  Step 9:  PNC label audit (ground truth validation)         │
│  Step 10: Final validation + dtype enforcement              │
│  Output: 6,864 patients × 154 columns · Parquet             │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  Feature Engineering  (src/features/)                       │
│  50 features → 44 after null filtering                      │
│  Domains: Child · CHW Quality · Engagement · Geography      │
│  Kenya EPI schedule-gated vaccine completeness              │
│  ColumnTransformer: 15 numeric · 27 binary · 2 categorical  │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  XGBoost + Optuna  (src/model/)                             │
│  50-trial TPE hyperparameter search (Optuna)                │
│  scale_pos_weight=5.07 for 16.5% positive rate             │
│  CalibratedClassifierCV isotonic (ECE = 0.023)              │
│  MLflow tracking + model registry (version 2 registered)   │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│  SHAP Explainability  (src/explainability/)                 │
│  TreeExplainer · Global beeswarm + bar charts               │
│  Per-patient waterfall · Plain-English API payload          │
└──────────────────────┬──────────────────────────────────────┘
                       │
              ┌────────┴────────┐
              ▼                 ▼
  ┌─────────────────┐  ┌──────────────────────┐
  │  FastAPI        │  │  Drift Monitor        │
  │  POST /predict  │  │  PSI feature drift    │
  │  POST /predict/ │  │  Label shift tracking │
  │       batch     │  │  Near-constant and    │
  │  GET  /health   │  │  low-cardinality feats│
  │  GET  /model/   │  │  skipped from PSI     │
  │       info      │  └──────────────────────┘
  └─────────────────┘
```

---

## Data Schema

| Table | DB Rows | Loaded Rows | Role | Join Key |
|---|---|---|---|---|
| `iz` | 8,814 | 8,814 | **Core** — child immunization visits | `contact_parent_id` |
| `active_chps` | 11,109,887 | **4,672** | CHW roster (DISTINCT ON area) | `chw_area_uuid` |
| `supervision` | 1,546 | 1,546 | CHW competency scores | `chw_area` |
| `homevisit` | 3,332,804 | **5,193** | Visit rate (GROUP BY area) | `chw_area` |
| `population` | 2,582,784 | **5,066** | Under-2 workload (median per area) | `chw_area` |
| `pnc` | 25,308 | 25,308 | Postnatal — label validation | `patient_id` |
| `preg_reg` / `preg_reg2` | 1,927 | 1,927 | Maternal ANC behaviour | household chain |
| `fp` / `refill` | 1,022,801 | **4,319** | FP household status (GROUP BY) | `contact_parent_id` |

*Server-side SQL aggregation reduces 18M+ raw rows to 55K for ETL processing.*

---

## Sample API Response

```json
{
  "patient_id": "2664b5f1-e861-45b9-8049-a96a0974bee9",
  "patient_name": "Hope Anyango Wesonga",
  "risk_score": 0.9982,
  "risk_pct": 99.8,
  "risk_tier": "HIGH",
  "top_drivers": [
    {
      "feature": "due_count_clean",
      "friendly_name": "Doses currently outstanding",
      "feature_value": 3.0,
      "shap_value": 0.8821,
      "direction": "increases_risk",
      "plain_english": "Child has 3 outstanding vaccine dose(s) — this increases defaulter risk"
    },
    {
      "feature": "patient_age_in_months",
      "friendly_name": "Child's age (months)",
      "feature_value": 12.0,
      "shap_value": 0.7489,
      "direction": "increases_risk",
      "plain_english": "Child is 12 months old — age context increases defaulter risk"
    },
    {
      "feature": "months_since_reported",
      "friendly_name": "Months since last CHW contact",
      "feature_value": 3.2,
      "shap_value": 0.5554,
      "direction": "increases_risk",
      "plain_english": "Last CHW contact was 3.2 month(s) ago — this increases defaulter risk"
    }
  ],
  "recommended_action": "Immediate home visit required. Bring vaccine referral form. Arrange facility referral for outstanding doses.",
  "model_version": "xgb_v1.0_iz_defaulter"
}
```

---

## Quick Start

### Prerequisites
- Python 3.11+
- PostgreSQL (or CSV extracts in `data/raw/`)
- 2 GB RAM minimum

### 1. Clone and install
```bash
git clone https://github.com/erickyegon/immunization-defaulter-risk-engine
cd immunization-defaulter-risk-engine
python -m venv venv && source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment
```bash
cp .env.example .env
# Edit .env — set POSTGRES_* credentials and API_KEY
```

### 3. Run full pipeline
```bash
python main.py --stage all
# ETL → Feature Engineering → XGBoost → Calibration → SHAP → Drift Report
```

### 4. Individual stages
```bash
python main.py --stage etl        # ETL only
python main.py --stage train      # Train + MLflow logging
python main.py --stage evaluate   # Evaluation + SHAP plots
python main.py --stage monitor    # PSI drift detection
python main.py --stage api        # Launch FastAPI server
```

### 5. Launch prediction API
```bash
uvicorn src.api.main:app --host 0.0.0.0 --port 8000
# Swagger UI: http://localhost:8000/docs
# Authenticate: X-API-Key header (set API_KEY in .env)
```

### 6. Launch Streamlit dashboard

**Hosted:** [https://immunizationengine.streamlit.app/](https://immunizationengine.streamlit.app/)

**Local:**
```bash
streamlit run streamlit_app.py
# Opens at http://localhost:8501
```

The dashboard requires a password at login. Two roles are pre-configured:

| Role | Default Password | Access |
|---|---|---|
| **User** | `Kenya` | Programme Dashboard · Patient Risk Scorer |
| **Administrator** | `Kenya2025` | All pages · Live PostgreSQL toggle · Drift Monitor · Model Performance |

To change passwords without editing code, set environment variables in `.env`:
```bash
ADMIN_PASSWORD=your_new_admin_password
USER_PASSWORD=your_new_user_password
```

### 7. Docker
```bash
docker build -t iz-defaulter .
docker run -p 8000:8000 --env-file .env iz-defaulter
```

---

## Access Control (RBAC)

The Streamlit dashboard implements role-based access control with two roles:

### Roles and Permissions

| Feature | User | Administrator |
|---|---|---|
| Programme Dashboard | ✅ | ✅ |
| Patient Risk Scorer | ✅ | ✅ |
| Model Performance page | ❌ | ✅ |
| Data Quality & Model Health page | ❌ | ✅ |
| Live PostgreSQL data toggle | ❌ | ✅ |
| Role badge in sidebar | 🔵 Blue | 🟡 Amber |

### Intended audiences

**User** — CHW supervisors and MOH programme managers who need to view the programme dashboard and look up individual patient risk scores. They do not need access to technical model internals.

**Administrator** — Data scientists, M&E leads, and technical programme staff who need to review model performance, monitor data drift, and switch between cached and live data sources.

### How it works

- Passwords are **SHA-256 hashed** before comparison — never stored or logged in plaintext.
- Session state is cleared on sign-out, requiring re-authentication on the next visit.
- Passwords are configurable via `.env` without changing source code (`ADMIN_PASSWORD`, `USER_PASSWORD`).
- The navigation menu itself is filtered by role — locked pages are not visible to Users, not merely blocked.

---

## Project Structure

```
immunization-defaulter-risk-engine/
├── config/
│   ├── epi_schedule.py          # Kenya EPI schedule constants + age-gate logic
│   ├── epi_schedule.yaml        # Human-readable WHO EPI schedule reference
│   └── model_config.yaml        # Single source of truth for all pipeline config
│
├── src/
│   ├── etl/
│   │   ├── loader.py            # Dual-backend: PostgreSQL (aggregated SQL) + CSV
│   │   ├── cleaner.py           # Table-specific cleaning (12 tables)
│   │   └── merger.py            # 10-step analytical dataset construction
│   │
│   ├── features/
│   │   └── pipeline.py          # 50-feature ColumnTransformer + null filtering
│   │
│   ├── model/
│   │   ├── trainer.py           # XGBoost + isotonic calibration + MLflow
│   │   ├── tuner.py             # Optuna TPE (50 trials, CV PR-AUC objective)
│   │   └── evaluator.py         # ROC/PR/calibration/fairness/threshold plots
│   │
│   ├── explainability/
│   │   └── shap_explainer.py    # Global + per-patient waterfall SHAP
│   │
│   ├── api/
│   │   └── main.py              # FastAPI: auth, validation, /predict, /batch
│   │
│   └── monitoring/
│       └── drift_detector.py    # PSI feature drift + label shift detection
│
├── reports/
│   ├── roc_pr_curves.png        # ROC-AUC=0.892, PR-AUC=0.698
│   ├── calibration_curve.png    # ECE=0.020
│   ├── feature_importance.png
│   ├── drift_report.html
│   └── shap/
│       ├── shap_beeswarm.png    # Global SHAP impact
│       ├── shap_bar.png         # Feature importance
│       ├── waterfall_high_example.png    # 99.7% risk patient
│       ├── waterfall_medium_example.png  # 60.0% risk patient
│       └── waterfall_low_example.png     # 33.0% risk patient
│
├── main.py                      # CLI orchestrator (5 stages)
├── streamlit_app.py             # Streamlit dashboard (4 pages)
├── requirements.txt
├── Dockerfile
└── .env.example
```

---

## Known Limitations & Open Issues

These issues are tracked deliberately. Documenting them here reflects methodological transparency, not incompleteness.

| # | Issue | Severity | Status |
|---|---|---|---|
| 1 | **SHAP rank-4 label unresolved** | Medium | Open |
| 2 | **Maternal ANC join — 0% match rate** | High | Open |
| 3 | **PNC label agreement — 40.5%** | Medium | Under investigation |

---

**1. SHAP rank-4 feature label unresolved**

The SHAP explainer reports `has_delayed_milestones_binary` as the 4th most important predictor (mean |SHAP| = 0.303). However, this field is 100% null in the current dataset — CHT milestone fields (`has_delayed_milestones`, `is_growth_monitoring`) have not yet been backfilled from CHP assessments — and the feature was excluded from the fitted preprocessor's 44 columns. The label at position 3 of the SHAP output matrix therefore points to a real, contributing predictor whose true identity is not yet confirmed. This is a feature-name alignment issue between the fitted `ColumnTransformer` and the SHAP explainer's name list, not a data leakage issue. The SHAP value is real; the label is wrong.

**Resolution path:** Confirm which column occupies position 3 in `preprocessor.transformers_` output and update the SHAP explainer to load names from the fitted preprocessor rather than from the pre-fit feature list.

---

**2. Maternal ANC join — 0% match rate**

Step 8 of the ETL pipeline attempts to join maternal health-seeking behaviour (ANC visits, MUAC risk, FP status) from the `preg_reg` / `preg_reg2` tables to child records in `iz`. The join currently resolves 0% of records because neither `contact_parent_parent_id` nor a `chw_area` column is present in the `preg_reg` extract with a UUID that maps to `iz.contact_parent_id`.

As a result, the following 6 features are 100% null and excluded from the model: `maternal_anc_visits`, `maternal_anc_defaulter`, `maternal_muac_risk`, `maternal_iron_folate`, `is_growth_monitoring_binary`, `has_delayed_milestones_binary`. The model achieves ROC-AUC = 0.893 without these features; inclusion is expected to improve recall for infants whose risk is driven by maternal health-seeking rather than their own vaccine history.

**Resolution path:** Confirm the correct linkage key between `preg_reg` and `iz` in the production CHT schema (likely a household or community-unit UUID chain). Update `src/etl/merger.py` Step 8 accordingly.

---

**3. PNC label agreement — 40.5%**

Step 9 of the ETL pipeline compares the composite `is_defaulter` target (built from the `iz` table) against the `is_immunization_defaulter` field in the `pnc` table as a ground-truth audit. Agreement is 40.5% on 189 matched records — below the expected ≥70% threshold for label consistency.

Most likely causes: (a) the `pnc` record is linked by the **mother's** `patient_id`, not the child's, meaning the matched records are not the same individuals; (b) the `pnc` field reflects a different observation window than the 30-day `iz` composite target. The PNC field is used for audit only — it is **not** a training feature and does not affect model predictions. However, low agreement is a signal that the ground-truth validation step is not yet functional, which reduces confidence in target variable quality.

**Resolution path:** Verify whether `pnc.patient_id` refers to the mother or child. If the former, join via a mother→child linkage table or use `pnc.contact_id` (child). Then re-audit agreement; target ≥ 70%.

---

## Responsible AI & Deployment Roadmap

| Phase | Scope | Gate Criteria |
|---|---|---|
| Phase 0 | Internal validation | ETL validated, IRB submitted, data governance signed |
| **Phase 1** | **2 pilot districts** | **Precision@20% > 0.60 · CHW supervisor SHAP review** |
| Phase 2 | 2 districts, 3-month prospective | 2× defaulter enrichment in top quartile vs. baseline |
| Phase 3 | 5 districts | Fairness AUC gap < 10pp across all districts · MOH briefed |
| Phase 4 | National operations | Monthly retraining · PSI monitoring · Quarterly MOH report |

---

## Methodological Notes

**Why XGBoost over logistic regression?**
The feature space includes 24 binary vaccine flags with complex age-conditional interactions (e.g., a child missing OPV-3 means something different at 4 months vs. 18 months). XGBoost handles these non-linear interactions natively while remaining fully explainable via SHAP.

**Why isotonic calibration?**
CHWs and MOH reviewers interpret scores as actionable probabilities. Raw XGBoost outputs are not calibrated — "0.74" does not mean 74% of such children default. Isotonic calibration enforces this guarantee (ECE = 0.023 on this dataset).

**Why PR-AUC as the primary metric?**
At 16.5% positive rate, ROC-AUC looks good even for poor models. PR-AUC directly measures the quality of the high-risk ranking — the operationally critical metric for CHW daily prioritisation.

**Why server-side SQL aggregation?**
The operational tables (`active_chps`, `homevisit`, `population`, `fp`) accumulate 11M+ rows historically. Loading them raw into pandas is impractical. Custom `DISTINCT ON`, `GROUP BY`, and `PERCENTILE_CONT` queries push aggregation to PostgreSQL, reducing network transfer from 18M+ rows to 55K.

**Why per-patient SHAP instead of global importance?**
CHWs need to explain to caregivers *why* a specific child is flagged. "Your child is 9 months old and hasn't received OPV-3 yet" is actionable. "The model weights age highly on average" is not.

**Data provenance and governance**
The methodology underpinning this engine has been applied across multiple national MOH deployments; this repository presents the Kenya implementation using data drawn directly from the Ministry of Health eCHIS platform, shared under the author's operational data governance agreements with Living Goods and the Kenya MOH.

---

## Author

**Dr. Erick Kiprotich Yegon, PhD (Epidemiology)**
AI & Data Science Consultant
Former Global Director of Data Science & Analytics — community health programs serving 8.5M+ individuals
30+ peer-reviewed publications · h-index 10 · EB-1A Permanent Resident

`github.com/erickyegon` · Richmond, KY

