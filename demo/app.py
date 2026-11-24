"""FastAPI demo server for PQC-TLS anomaly detection."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = Path(__file__).resolve().parents[1]
ARTIFACTS = ROOT / "artifacts"
STATIC = Path(__file__).resolve().parent / "static"

app = FastAPI(
    title="PQC-TLS Sentinel",
    description="Explainable anomaly detection demo for hybrid Post-Quantum TLS 1.3",
    version="1.0.0",
)

STATE: dict[str, Any] = {}


def _load_artifacts() -> None:
    summary_path = ARTIFACTS / "summary.json"
    sessions_path = ARTIFACTS / "demo_sessions.json"
    if not summary_path.exists():
        raise RuntimeError(
            "Missing artifacts. Run: python -m src.train_and_export"
        )

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    sessions = json.loads(sessions_path.read_text(encoding="utf-8"))
    models = {}
    for name in summary["models"]:
        safe = name.lower().replace(" ", "_")
        models[name] = joblib.load(ARTIFACTS / f"model_{safe}.joblib")

    STATE["summary"] = summary
    STATE["sessions"] = sessions
    STATE["models"] = models
    STATE["feature_columns"] = summary["feature_columns"]
    STATE["thresholds"] = summary["thresholds"]
    STATE["explainers"] = {}

    # Lazy SHAP explainers for tree models only
    try:
        import shap

        for name in ("Random Forest", "XGBoost"):
            if name in models:
                clf = models[name].named_steps["classifier"]
                STATE["explainers"][name] = shap.TreeExplainer(clf)
    except Exception as exc:  # noqa: BLE001
        STATE["shap_error"] = str(exc)


@app.on_event("startup")
def startup() -> None:
    _load_artifacts()


class PredictRequest(BaseModel):
    features: dict[str, float | bool | None] = Field(
        ..., description="Feature name → value map. Null allowed for missing."
    )
    model: str | None = Field(
        default=None, description="Model name. Defaults to best validation model."
    )


def _row_from_features(features: dict) -> pd.DataFrame:
    cols = STATE["feature_columns"]
    row = {}
    for c in cols:
        v = features.get(c, None)
        if v is None or (isinstance(v, float) and np.isnan(v)):
            row[c] = np.nan
        else:
            row[c] = float(v)
    return pd.DataFrame([row], columns=cols)


def _top_contributions(model_name: str, X: pd.DataFrame, top_k: int = 10) -> list[dict]:
    pipeline = STATE["models"][model_name]
    imputer = pipeline.named_steps["imputer"]
    transformed = imputer.transform(X)
    feature_names = list(imputer.get_feature_names_out(STATE["feature_columns"]))

    if model_name in STATE.get("explainers", {}):
        explainer = STATE["explainers"][model_name]
        sv = explainer.shap_values(transformed)
        if isinstance(sv, list):
            values = np.asarray(sv[1][0])
        else:
            values = np.asarray(sv[0])
            if values.ndim == 2:
                values = values[:, 1] if values.shape[1] == 2 else values.ravel()
        pairs = list(zip(feature_names, values.tolist()))
    elif model_name == "Logistic Regression":
        scaler = pipeline.named_steps["scaler"]
        clf = pipeline.named_steps["classifier"]
        scaled = scaler.transform(transformed)
        contrib = (scaled[0] * clf.coef_[0]).tolist()
        pairs = list(zip(feature_names, contrib))
    else:
        return []

    pairs.sort(key=lambda t: abs(t[1]), reverse=True)
    out = []
    for name, value in pairs[:top_k]:
        out.append(
            {
                "feature": name,
                "contribution": float(value),
                "direction": "anomaly" if value > 0 else "normal",
            }
        )
    return out


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "artifacts_loaded": bool(STATE),
        "best_model": STATE.get("summary", {}).get("best_model"),
    }


@app.get("/api/summary")
def summary():
    if not STATE:
        raise HTTPException(503, "Artifacts not loaded")
    return STATE["summary"]


@app.get("/api/sessions")
def sessions(tag: str | None = None, limit: int = 80):
    items = STATE["sessions"]
    if tag:
        items = [s for s in items if s["tag"] == tag]
    return {"count": len(items[:limit]), "sessions": items[:limit]}


@app.get("/api/sessions/{session_id}")
def session_detail(session_id: str):
    for s in STATE["sessions"]:
        if s["id"] == session_id:
            return s
    raise HTTPException(404, "Session not found in demo sample pool")


@app.post("/api/predict")
def predict(body: PredictRequest):
    if not STATE:
        raise HTTPException(503, "Artifacts not loaded")

    model_name = body.model or STATE["summary"]["best_model"]
    if model_name not in STATE["models"]:
        raise HTTPException(400, f"Unknown model: {model_name}")

    X = _row_from_features(body.features)
    model = STATE["models"][model_name]
    threshold = float(STATE["thresholds"][model_name])
    proba = float(model.predict_proba(X)[0, 1])
    pred = int(proba >= threshold)

    contributions = _top_contributions(model_name, X)

    return {
        "model": model_name,
        "threshold": threshold,
        "anomaly_probability": proba,
        "prediction": pred,
        "prediction_label": "Anomaly" if pred == 1 else "Normal",
        "contributions": contributions,
        "note": (
            "Feature contributions are SHAP values for tree models "
            "(or coefficient×value for logistic regression). "
            "They explain the model, not causal proof of an attack."
        ),
    }


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
