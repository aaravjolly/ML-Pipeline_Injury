"""Tests for the FastAPI service. Skipped if FastAPI isn't installed."""

import os
import warnings
from pathlib import Path

import pytest

warnings.filterwarnings("ignore", category=FutureWarning, module="sklearn")

fastapi = pytest.importorskip("fastapi")
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    # Train a tiny model and point the API at it.
    import joblib

    from src.data import (
        SyntheticConfig,
        athlete_holdout_split,
        generate_synthetic_athletes,
    )
    from src.evaluation import evaluate
    from src.models import build_pipeline

    df = generate_synthetic_athletes(SyntheticConfig(n_athletes=40, n_weeks=12, seed=0))
    tr, te = athlete_holdout_split(df, test_frac=0.2, seed=0)
    y_tr = tr["injured"].astype(int).to_numpy()
    y_te = te["injured"].astype(int).to_numpy()
    pipeline = build_pipeline(model_kind="logreg", random_state=0)
    pipeline.fit(tr, y_tr)
    proba_te = pipeline.predict_proba(te)[:, 1]
    metrics = evaluate(y_te, proba_te).as_dict()

    tmp = tmp_path_factory.mktemp("models")
    path = tmp / "pipeline.joblib"
    joblib.dump({
        "pipeline": pipeline,
        "metadata": {
            "model_kind": "logreg",
            "best_threshold": 0.3,
            "test_metrics": metrics,
        },
    }, path)
    os.environ["INJURY_MODEL_PATH"] = str(path)

    from src.api import create_app
    return TestClient(create_app())


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


def _record(week=10, weekly_load=2200, sleep=6.5, soreness=6):
    """Build a valid AthleteWeek record for tests."""
    return {
        "athlete_id": "A0001",
        "week": week,
        "age": 24.0,
        "weekly_load": weekly_load,
        "sleep_hours": sleep,
        "soreness": soreness,
        "rpe_avg": 7.0,
        "sessions_count": 5,
        "prior_injuries": 0,
    }


class TestAPIEndpoints:
    def test_health(self, client):
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"

    def test_info(self, client):
        r = client.get("/info")
        assert r.status_code == 200
        body = r.json()
        assert "model_kind" in body
        assert "metadata" in body

    def test_predict_one(self, client):
        r = client.post("/predict", json={
            "target": _record(),
            "history": [],
            "threshold": 0.5,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["athlete_id"] == "A0001"
        assert body["week"] == 10
        assert 0.0 <= body["risk_probability"] <= 1.0
        assert body["predicted_injured"] in {0, 1}

    def test_predict_with_history(self, client):
        history = [_record(week=w) for w in range(7, 10)]
        r = client.post("/predict", json={
            "target": _record(week=10),
            "history": history,
            "threshold": 0.5,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["week"] == 10

    def test_predict_high_risk_inputs_higher(self, client):
        """Heavy load + low sleep + high soreness should give higher risk
        than the opposite, on average."""
        high = _record(weekly_load=3500, sleep=4.0, soreness=10)
        low = _record(weekly_load=1200, sleep=8.5, soreness=2)
        r_high = client.post("/predict", json={
            "target": high, "history": [], "threshold": 0.5,
        }).json()
        r_low = client.post("/predict", json={
            "target": low, "history": [], "threshold": 0.5,
        }).json()
        assert r_high["risk_probability"] > r_low["risk_probability"]

    def test_predict_batch(self, client):
        records = [_record(week=w) for w in range(5)]
        r = client.post("/predict/batch", json={
            "records": records,
            "threshold": 0.5,
        })
        assert r.status_code == 200
        body = r.json()
        assert body["n"] == 5
        assert len(body["results"]) == 5
        for res in body["results"]:
            assert 0.0 <= res["risk_probability"] <= 1.0

    def test_predict_validation_errors(self, client):
        # Soreness out of range.
        bad = _record()
        bad["soreness"] = 99
        r = client.post("/predict", json={
            "target": bad, "history": [], "threshold": 0.5,
        })
        assert r.status_code == 422

    def test_root_html(self, client):
        r = client.get("/")
        assert r.status_code == 200
        assert "injury risk" in r.text.lower()
