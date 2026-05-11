"""
FastAPI service for injury-risk prediction.

Endpoints
---------
GET  /health             Liveness check.
GET  /info               Pipeline metadata + training metrics.
POST /predict            Predict for a single athlete-week record.
POST /predict/batch      Predict for many records at once.
GET  /                   Tiny browser console.

The trained pipeline (Pipeline = FeatureEngineer -> Preprocessor ->
Classifier) is loaded from a single joblib file at startup. The path
defaults to ``models/pipeline.joblib`` and can be overridden with
``INJURY_MODEL_PATH``.

Predictions require some week-by-week history per athlete to compute
the rolling features (ACWR, soreness windows, etc.). If you only send
the single most-recent week, you'll still get a prediction but the
rolling features will be defaulted to that week's values - the model
handles this case via the median imputer in the preprocessor.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Pydantic schemas
# ---------------------------------------------------------------------------


class AthleteWeek(BaseModel):
    """One athlete-week record. Required fields match the training schema."""

    athlete_id: str
    week: int
    age: float
    weekly_load: float
    sleep_hours: float
    soreness: int = Field(..., ge=1, le=10)

    # Optional fields - imputed if missing.
    sex: Optional[str] = None
    position: Optional[str] = None
    rpe_avg: Optional[float] = None
    sessions_count: Optional[int] = None
    sprint_distance_km: Optional[float] = None
    prior_injuries: Optional[int] = None
    # Label is unknown at predict time, but the FeatureEngineer needs the
    # column to compute injury_history_4w. Default to 0; the label won't
    # be used for prediction since FeatureEngineer drops it.
    injured: int = 0


class PredictRequest(BaseModel):
    """A single prediction. Optionally include prior weeks for rolling features."""

    target: AthleteWeek
    history: List[AthleteWeek] = Field(default_factory=list,
                                        description="Prior weeks for the same "
                                                    "athlete (will be merged "
                                                    "before feature engineering)")
    threshold: float = 0.5


class BatchPredictRequest(BaseModel):
    records: List[AthleteWeek]
    threshold: float = 0.5


class PredictResponse(BaseModel):
    athlete_id: str
    week: int
    risk_probability: float
    predicted_injured: int
    threshold: float


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def _resolve_model_path() -> Path:
    p = os.environ.get("INJURY_MODEL_PATH")
    if p:
        return Path(p)
    return Path(__file__).resolve().parent.parent / "models" / "pipeline.joblib"


def create_app() -> FastAPI:
    model_path = _resolve_model_path()
    if not model_path.exists():
        raise RuntimeError(
            f"No trained model at {model_path}. "
            "Run scripts/train.py first."
        )
    bundle = joblib.load(model_path)
    pipeline = bundle["pipeline"]
    metadata = bundle.get("metadata", {})

    app = FastAPI(
        title="Injury Risk Predictor",
        version="1.0.0",
        description="ML pipeline that predicts athlete injury risk for the "
                    "current week from training, sleep, and history features.",
    )
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"],
        allow_methods=["*"], allow_headers=["*"],
    )

    # ------------------------------------------------------------------
    @app.get("/health")
    def health() -> Dict:
        return {"status": "ok"}

    @app.get("/info")
    def info() -> Dict:
        return {
            "model_path": str(model_path),
            "metadata": metadata,
            "model_kind": str(type(pipeline.named_steps["classifier"]).__name__),
        }

    # ------------------------------------------------------------------
    @app.post("/predict", response_model=PredictResponse)
    def predict_one(req: PredictRequest) -> PredictResponse:
        # Build a frame from history + target, sorted, ready for the FE.
        df = _records_to_frame([*req.history, req.target])
        try:
            proba = pipeline.predict_proba(df)[:, 1]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"prediction failed: {exc}") from exc
        # The target row is the one we want to score - find it by (id, week).
        mask = (df["athlete_id"] == req.target.athlete_id) & (df["week"] == req.target.week)
        if not mask.any():
            raise HTTPException(500, "target row not found after frame assembly")
        idx = int(np.argmax(mask.values))
        risk = float(proba[idx])
        return PredictResponse(
            athlete_id=req.target.athlete_id,
            week=req.target.week,
            risk_probability=risk,
            predicted_injured=int(risk >= req.threshold),
            threshold=req.threshold,
        )

    @app.post("/predict/batch")
    def predict_batch(req: BatchPredictRequest) -> Dict:
        if not req.records:
            return {"results": []}
        df = _records_to_frame(req.records)
        try:
            proba = pipeline.predict_proba(df)[:, 1]
        except Exception as exc:  # noqa: BLE001
            raise HTTPException(500, f"prediction failed: {exc}") from exc

        results = []
        # The frame may have been re-sorted by athlete_id/week; we need to
        # map probabilities back to the input order.
        df_pos = (
            df.assign(_pos=range(len(df)))
              .set_index(["athlete_id", "week"])
        )
        for r in req.records:
            try:
                pos = int(df_pos.loc[(r.athlete_id, r.week), "_pos"])
            except KeyError:
                raise HTTPException(500, f"missing prediction for {r.athlete_id}/{r.week}")
            p = float(proba[pos])
            results.append({
                "athlete_id": r.athlete_id,
                "week": r.week,
                "risk_probability": p,
                "predicted_injured": int(p >= req.threshold),
                "threshold": req.threshold,
            })
        return {"results": results, "n": len(results)}

    @app.get("/", response_class=HTMLResponse)
    def root() -> str:
        return _CONSOLE_HTML

    return app


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _records_to_frame(records: List[AthleteWeek]) -> pd.DataFrame:
    """Convert pydantic records to the DataFrame the FE expects."""
    if not records:
        raise HTTPException(400, "no records provided")
    rows = [r.model_dump() for r in records]
    df = pd.DataFrame(rows)
    return df


# ---------------------------------------------------------------------------
# Tiny browser console
# ---------------------------------------------------------------------------


_CONSOLE_HTML = """<!doctype html>
<html><head><meta charset="utf-8" /><title>injury risk</title>
<style>
  :root { --bg:#0c0e12; --panel:#14171d; --line:#1f242c; --ink:#e8eaed; --dim:#7d8590; --accent:#ffe066; --warn:#ff5c5c; --ok:#62d99c; }
  *{box-sizing:border-box;margin:0;padding:0}
  body{font-family:ui-monospace,Menlo,monospace;background:var(--bg);color:var(--ink);padding:32px;max-width:760px;margin:0 auto;line-height:1.5}
  h1{font-size:22px;margin-bottom:6px}
  .sub{color:var(--dim);font-size:13px;margin-bottom:24px}
  label{display:block;font-size:11px;text-transform:uppercase;letter-spacing:.12em;color:var(--dim);margin:12px 0 4px}
  input{width:100%;padding:9px;background:var(--panel);border:1px solid var(--line);color:var(--ink);font-family:inherit}
  input:focus{outline:none;border-color:var(--accent)}
  .grid{display:grid;grid-template-columns:1fr 1fr;gap:14px}
  button{margin-top:18px;padding:11px 22px;background:var(--accent);color:black;border:0;font-weight:700;cursor:pointer;text-transform:uppercase;letter-spacing:.1em;font-size:11px}
  button:hover{background:white}
  .result{margin-top:24px;padding:18px;background:var(--panel);border:1px solid var(--line)}
  .risk{font-size:34px;font-weight:bold;font-family:'Fraunces',serif}
  .risk.low{color:var(--ok)} .risk.med{color:var(--accent)} .risk.high{color:var(--warn)}
  .meta{color:var(--dim);font-size:12px;margin-top:6px}
</style></head><body>
<h1>injury risk</h1>
<div class="sub">predict next-week injury probability from training load and recovery</div>
<div class="grid">
  <div><label>athlete id</label><input id="aid" value="A0001" /></div>
  <div><label>week</label><input id="week" type="number" value="10" /></div>
  <div><label>age</label><input id="age" type="number" step="0.5" value="24" /></div>
  <div><label>weekly load</label><input id="load" type="number" step="50" value="2200" /></div>
  <div><label>sleep hours</label><input id="sleep" type="number" step="0.1" value="6.5" /></div>
  <div><label>soreness (1-10)</label><input id="soreness" type="number" min="1" max="10" value="6" /></div>
  <div><label>rpe avg</label><input id="rpe" type="number" step="0.1" value="7.5" /></div>
  <div><label>prior injuries</label><input id="prior" type="number" value="1" /></div>
</div>
<button id="go">predict</button>
<div id="out" class="result" style="display:none"></div>
<script>
async function go() {
  const target = {
    athlete_id: document.getElementById('aid').value,
    week: +document.getElementById('week').value,
    age: +document.getElementById('age').value,
    weekly_load: +document.getElementById('load').value,
    sleep_hours: +document.getElementById('sleep').value,
    soreness: +document.getElementById('soreness').value,
    rpe_avg: +document.getElementById('rpe').value,
    prior_injuries: +document.getElementById('prior').value,
  };
  const r = await fetch('/predict', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({target, history: [], threshold: 0.5}),
  });
  const data = await r.json();
  const out = document.getElementById('out');
  out.style.display = 'block';
  const pct = (data.risk_probability * 100).toFixed(1);
  let cls = 'low'; if (data.risk_probability > 0.15) cls = 'med'; if (data.risk_probability > 0.30) cls = 'high';
  out.innerHTML = '<div class="risk ' + cls + '">' + pct + '%</div>'
    + '<div class="meta">predicted: ' + (data.predicted_injured ? 'injured' : 'healthy')
    + ' &middot; threshold ' + data.threshold + '</div>';
}
document.getElementById('go').onclick = go;
</script></body></html>
"""


# Lazy app construction: tests can monkey-patch the model path.
try:
    app = create_app()
except Exception as _exc:  # noqa: BLE001
    app = FastAPI(title="Injury Risk (no model)")

    @app.get("/health")
    def _h() -> Dict:
        return {"status": "no_model", "error": str(_exc)}
