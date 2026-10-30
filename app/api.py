"""Prediction API.  Run:  make serve   (Swagger page at http://localhost:8000/docs)"""

import json
import logging
import sys
import uuid
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import ROOT  # noqa: E402
from src.serving import Predictor  # noqa: E402

LOG_DIR = ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
log = logging.getLogger("port-cargo")
logging.basicConfig(level=logging.INFO, format="%(message)s")

app = FastAPI(
    title="Sri Lanka Port Cargo Estimator",
    description="EBM estimate of daily import cargo (tonnes) with an exact explanation.",
    version="1.0.0",
)
predictor = Predictor()


class PredictRequest(BaseModel):
    port: Literal["Colombo", "Hambantota", "Trincomalee", "Galle"]
    date: date
    portcalls_container: int = Field(0, ge=0, le=60)
    portcalls_dry_bulk: int = Field(0, ge=0, le=60)
    portcalls_general_cargo: int = Field(0, ge=0, le=60)
    portcalls_roro: int = Field(0, ge=0, le=60)
    portcalls_tanker: int = Field(0, ge=0, le=60)
    import_roll7: float | None = Field(None, ge=0, description="Auto-filled from history if empty")
    portcalls_roll7: float | None = Field(None, ge=0, description="Auto-filled if empty")

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "port": "Colombo",
                    "date": "2026-07-25",
                    "portcalls_container": 10,
                    "portcalls_dry_bulk": 1,
                    "portcalls_general_cargo": 0,
                    "portcalls_roro": 1,
                    "portcalls_tanker": 1,
                }
            ]
        }
    }


class Feedback(BaseModel):
    request_id: str
    actual_import_tonnes: float = Field(ge=0)


def _event(kind: str, payload: dict):
    record = {"event": kind, "timestamp": datetime.now(UTC).isoformat(), **payload}
    line = json.dumps(record, default=str)
    log.info(line)
    with open(LOG_DIR / f"{kind}.jsonl", "a") as f:
        f.write(line + "\n")


@app.get("/health")
def health():
    return {"status": "healthy", "model_version": predictor.version}


@app.get("/version")
def version():
    return {"model_version": predictor.version}


@app.post("/predict")
def predict(req: PredictRequest):
    calls = req.model_dump(
        include={
            "portcalls_container",
            "portcalls_dry_bulk",
            "portcalls_general_cargo",
            "portcalls_roro",
            "portcalls_tanker",
        }
    )
    result = predictor.predict(req.port, req.date, calls, req.import_roll7, req.portcalls_roll7)
    result["request_id"] = str(uuid.uuid4())
    _event(
        "prediction",
        {
            "request_id": result["request_id"],
            "inputs": result["inputs"],
            "prediction": result["prediction_tonnes"],
            "model_version": predictor.version,
        },
    )
    return result


@app.post("/feedback")
def feedback(fb: Feedback):
    """Send the real tonnage later; it becomes labelled data for monitoring and retraining."""
    _event("feedback", fb.model_dump())
    return {"status": "ok"}


@app.get("/explain/global")
def explain_global():
    return {"terms": predictor.global_importance()}


@app.get("/explain/shape/{feature}")
def explain_shape(feature: str):
    if feature not in predictor.model.term_names_ or " & " in feature:
        raise HTTPException(404, f"unknown feature '{feature}'")
    return predictor.shape_function(feature)


@app.get("/metrics")
def metrics():
    return predictor.metrics["eval_metrics"]
