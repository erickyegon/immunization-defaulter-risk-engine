"""
api/main.py
─────────────────────────────────────────────────────────────────────────────
FastAPI REST API for immunization defaulter risk prediction.
Endpoints:
  POST /predict          — single patient prediction with SHAP
  POST /predict/batch    — batch patient predictions
  GET  /health           — model readiness check
  GET  /model/info       — model metadata

Usage:
  cd iz_defaulter_model
  uvicorn src.api.main:app --host 0.0.0.0 --port 8000
"""

import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
import yaml
from fastapi import Depends, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field, model_validator

logger = logging.getLogger(__name__)

# ── API key authentication ────────────────────────────────────────────────────

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Security(_API_KEY_HEADER)) -> None:
    expected = os.getenv("API_KEY", "").strip()
    if not expected:
        # No key configured — allow all (dev/local mode)
        return
    if api_key != expected:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


# ── Model state (loaded once at startup) ─────────────────────────────────────

STATE: Dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts at startup, clean up at shutdown."""
    logger.info("Loading model artifacts...")
    try:
        processed_dir = Path(os.getenv("MODEL_DIR", "data/processed"))
        STATE["model"]         = joblib.load(processed_dir / "model.pkl")
        STATE["preprocessor"]  = joblib.load(processed_dir / "preprocessor.pkl")
        STATE["feature_names"] = joblib.load(processed_dir / "feature_names.pkl")

        with open("config/model_config.yaml") as f:
            STATE["cfg"] = yaml.safe_load(f)

        STATE["ready"] = True
        logger.info("Model loaded successfully")
    except FileNotFoundError as e:
        logger.error(f"Model not trained yet: {e}")
        STATE["ready"] = False

    yield  # App runs here
    STATE.clear()


# ── CORS ──────────────────────────────────────────────────────────────────────

def _get_allowed_origins() -> List[str]:
    raw = os.getenv("CORS_ORIGINS", "").strip()
    if raw:
        return [o.strip() for o in raw.split(",") if o.strip()]
    # Default: localhost only (set CORS_ORIGINS in production)
    return ["http://localhost:3000", "http://localhost:8080"]


app = FastAPI(
    title       = "IZ Defaulter Prediction API",
    description = "XGBoost + SHAP immunization defaulter risk scoring for CHW platform",
    version     = "1.0.0",
    lifespan    = lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins  = _get_allowed_origins(),
    allow_methods  = ["POST", "GET"],
    allow_headers  = ["Content-Type", "X-API-Key"],
)


# ── Request / Response schemas ────────────────────────────────────────────────

class PatientFeatures(BaseModel):
    """Raw patient features — mirrors the analytical dataset columns."""
    patient_id:               str
    patient_name:             Optional[str] = None
    patient_age_in_months:    Optional[float] = Field(None, ge=0, le=60)
    patient_sex_binary:       Optional[float] = Field(None, ge=0, le=1)
    vax_completeness_score:   Optional[float] = Field(None, ge=0, le=1)
    vax_completeness_all:     Optional[float] = Field(None, ge=0, le=1)
    age_expected_vaccine_count: Optional[float] = Field(None, ge=0, le=20)
    due_count_clean:          Optional[float] = Field(None, ge=0, le=20)
    is_malaria_endemic_binary: Optional[float] = Field(None, ge=0, le=1)
    vitamin_a_completeness:   Optional[float] = Field(None, ge=0, le=1)
    is_growth_monitoring_binary: Optional[float] = Field(None, ge=0, le=1)
    has_delayed_milestones_binary: Optional[float] = Field(None, ge=0, le=1)
    penta_series_complete:    Optional[float] = Field(None, ge=0, le=1)
    opv_series_complete:      Optional[float] = Field(None, ge=0, le=1)
    pcv_series_complete:      Optional[float] = Field(None, ge=0, le=1)
    rota_series_complete:     Optional[float] = Field(None, ge=0, le=1)
    measles_booster_gap:      Optional[float] = Field(None, ge=-24, le=60)
    months_since_reported:    Optional[float] = Field(None, ge=0, le=120)
    has_bcg:                  Optional[float] = Field(None, ge=0, le=1)
    has_opv_0:                Optional[float] = Field(None, ge=0, le=1)
    has_opv_1:                Optional[float] = Field(None, ge=0, le=1)
    has_opv_2:                Optional[float] = Field(None, ge=0, le=1)
    has_opv_3:                Optional[float] = Field(None, ge=0, le=1)
    has_pcv_1:                Optional[float] = Field(None, ge=0, le=1)
    has_pcv_2:                Optional[float] = Field(None, ge=0, le=1)
    has_pcv_3:                Optional[float] = Field(None, ge=0, le=1)
    has_penta_1:              Optional[float] = Field(None, ge=0, le=1)
    has_penta_2:              Optional[float] = Field(None, ge=0, le=1)
    has_penta_3:              Optional[float] = Field(None, ge=0, le=1)
    has_ipv:                  Optional[float] = Field(None, ge=0, le=1)
    has_rota_1:               Optional[float] = Field(None, ge=0, le=1)
    has_rota_2:               Optional[float] = Field(None, ge=0, le=1)
    has_rota_3:               Optional[float] = Field(None, ge=0, le=1)
    has_measles_9_months:     Optional[float] = Field(None, ge=0, le=1)
    has_measles_18_months:    Optional[float] = Field(None, ge=0, le=1)
    chw_supervision_frequency: Optional[float] = Field(None, ge=0)
    chw_immunization_competency_pct: Optional[float] = Field(None, ge=0, le=100)
    chw_overall_assessment_pct: Optional[float] = Field(None, ge=0, le=100)
    chw_workload_u2:          Optional[float] = Field(None, ge=0)
    monthly_homevisit_rate:   Optional[float] = Field(None, ge=0)
    months_since_last_supervision: Optional[float] = Field(None, ge=0)
    chw_has_all_tools:        Optional[float] = Field(None, ge=0, le=1)
    chw_has_ppe:              Optional[float] = Field(None, ge=0, le=1)
    maternal_anc_visits:      Optional[float] = Field(None, ge=0, le=20)
    maternal_anc_defaulter:   Optional[float] = Field(None, ge=0, le=1)
    maternal_muac_risk:       Optional[float] = Field(None, ge=0, le=1)
    maternal_iron_folate:     Optional[float] = Field(None, ge=0, le=1)
    household_on_fp:          Optional[float] = Field(None, ge=0, le=1)
    sub_county_encoded:       Optional[float] = Field(None, ge=0)
    county_encoded:           Optional[float] = Field(None, ge=0)

    model_config = {"extra": "ignore"}


class SHAPDriver(BaseModel):
    feature:       str
    friendly_name: str
    feature_value: float
    shap_value:    float
    direction:     str
    plain_english: str


class PredictionResponse(BaseModel):
    patient_id:          str
    patient_name:        Optional[str]
    risk_score:          float = Field(description="Calibrated probability [0-1]")
    risk_pct:            float = Field(description="Risk as percentage")
    risk_tier:           str   = Field(description="LOW | MEDIUM | HIGH")
    top_drivers:         List[SHAPDriver]
    recommended_action:  str
    model_version:       str


class BatchRequest(BaseModel):
    patients: List[PatientFeatures]


class BatchResponse(BaseModel):
    predictions: List[PredictionResponse]
    n_high:      int
    n_medium:    int
    n_low:       int


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.get("/health")
async def health():
    return {
        "status": "ready" if STATE.get("ready") else "model_not_loaded",
        "model_version": "xgb_v1.0_iz_defaulter",
    }


@app.get("/model/info", dependencies=[Depends(verify_api_key)])
async def model_info():
    if not STATE.get("ready"):
        raise HTTPException(503, "Model not loaded")
    return {
        "model_type":        "XGBClassifier (CalibratedClassifierCV)",
        "n_features":        len(STATE["feature_names"]),
        "feature_names":     STATE["feature_names"],
        "risk_tiers":        STATE["cfg"]["api"]["risk_tiers"],
        "positive_label":    "immunization_defaulter",
    }


@app.post("/predict", response_model=PredictionResponse, dependencies=[Depends(verify_api_key)])
async def predict(patient: PatientFeatures):
    if not STATE.get("ready"):
        raise HTTPException(503, "Model not loaded — run training pipeline first")

    model        = STATE["model"]
    preprocessor = STATE["preprocessor"]
    feat_names   = STATE["feature_names"]
    cfg          = STATE["cfg"]

    # Build feature row
    row_dict = {k: v for k, v in patient.model_dump().items()
                if k not in ("patient_id", "patient_name")}
    X_raw = pd.DataFrame([row_dict])[feat_names].reindex(columns=feat_names)

    # Preprocess
    X_t = preprocessor.transform(X_raw)

    # Risk score
    risk_score = float(model.predict_proba(X_t)[0, 1])
    risk_tier  = _tier(risk_score, cfg["api"]["risk_tiers"])

    # SHAP
    drivers = _shap_drivers(model, X_t, feat_names, cfg["api"]["top_shap_drivers"])

    return PredictionResponse(
        patient_id         = patient.patient_id,
        patient_name       = patient.patient_name,
        risk_score         = round(risk_score, 4),
        risk_pct           = round(risk_score * 100, 1),
        risk_tier          = risk_tier,
        top_drivers        = drivers,
        recommended_action = _recommend(risk_tier, drivers),
        model_version      = "xgb_v1.0_iz_defaulter",
    )


@app.post("/predict/batch", response_model=BatchResponse, dependencies=[Depends(verify_api_key)])
async def predict_batch(req: BatchRequest):
    if not STATE.get("ready"):
        raise HTTPException(503, "Model not loaded")

    predictions = []
    for patient in req.patients:
        pred = await predict(patient)
        predictions.append(pred)

    return BatchResponse(
        predictions = predictions,
        n_high      = sum(1 for p in predictions if p.risk_tier == "HIGH"),
        n_medium    = sum(1 for p in predictions if p.risk_tier == "MEDIUM"),
        n_low       = sum(1 for p in predictions if p.risk_tier == "LOW"),
    )


# ── Helpers ───────────────────────────────────────────────────────────────────

def _tier(score: float, tiers: dict) -> str:
    if score < tiers["low"][1]:
        return "LOW"
    if score < tiers["medium"][1]:
        return "MEDIUM"
    return "HIGH"


def _shap_drivers(model, X_t: np.ndarray, feat_names, top_n: int) -> List[SHAPDriver]:
    """Compute SHAP values for a single prediction."""
    import shap
    from src.explainability.shap_explainer import SHAPExplainer

    xgb = SHAPExplainer._unwrap_model(model)
    try:
        explainer = shap.TreeExplainer(xgb)
        sv = explainer.shap_values(X_t)[0]
    except Exception:
        logger.exception("SHAP explanation failed — returning empty drivers")
        return []

    top_idx = np.argsort(np.abs(sv))[::-1][:top_n]
    drivers = []
    for i in top_idx:
        fname = feat_names[i] if i < len(feat_names) else f"f{i}"
        fval  = float(X_t[0, i])
        sval  = float(sv[i])
        drivers.append(SHAPDriver(
            feature       = fname,
            friendly_name = SHAPExplainer.FEATURE_LABELS.get(fname, fname),
            feature_value = round(fval, 4),
            shap_value    = round(sval, 4),
            direction     = "increases_risk" if sval > 0 else "decreases_risk",
            plain_english = f"{'Increases' if sval > 0 else 'Reduces'} risk (value={fval:.2f})",
        ))
    return drivers


def _recommend(tier: str, drivers: List[SHAPDriver]) -> str:
    base = {
        "LOW":    "Routine follow-up at next scheduled visit.",
        "MEDIUM": "Prioritise within-month home visit. Review vaccine card.",
        "HIGH":   "Immediate home visit required. Bring vaccine referral form.",
    }[tier]
    if tier == "HIGH" and drivers:
        feat = drivers[0].feature
        extras = {
            "measles_booster_gap": " Schedule MR2 booster referral.",
            "due_count_clean":     " Arrange facility referral for outstanding doses.",
        }
        base += extras.get(feat, "")
    return base
