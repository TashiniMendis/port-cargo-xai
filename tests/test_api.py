"""API tests run only when a trained model is present (after `dvc pull` or `make repro`)."""

import pytest

from src.config import MODELS, PROCESSED

pytestmark = pytest.mark.skipif(
    not (MODELS / "model.pkl").exists() or not (PROCESSED / "history.csv").exists(),
    reason="no trained model - run `uv run dvc pull` or `make repro`",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    from app.api import app

    return TestClient(app)


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "healthy"


def test_predict_returns_explanation_that_adds_up(client):
    body = {
        "port": "Colombo",
        "date": "2026-07-25",
        "portcalls_container": 10,
        "portcalls_dry_bulk": 1,
        "portcalls_tanker": 1,
    }
    r = client.post("/predict", json=body).json()
    assert r["prediction_tonnes"] > 0
    total = r["intercept"] + sum(t["value"] for t in r["contributions"])
    assert total == pytest.approx(r["raw_prediction"], abs=5)
    assert r["recent_activity_autofilled"] is True


def test_no_ships_gives_little_cargo(client):
    r = client.post("/predict", json={"port": "Galle", "date": "2026-07-25"}).json()
    assert r["prediction_tonnes"] < 1000


def test_invalid_port_rejected(client):
    assert client.post("/predict", json={"port": "Jaffna", "date": "2026-07-25"}).status_code == 422


def test_feedback(client):
    r = client.post("/feedback", json={"request_id": "abc", "actual_import_tonnes": 5000})
    assert r.json() == {"status": "ok"}
