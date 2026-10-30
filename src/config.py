"""Paths, feature lists and parameters shared by every stage."""

from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
MODELS = ROOT / "models"
METRICS = ROOT / "metrics"
PLOTS = ROOT / "plots"

TARGET = "import"  # daily import cargo, metric tonnes
PORTS = ["Colombo", "Hambantota", "Trincomalee", "Galle"]
VESSEL_TYPES = [
    "portcalls_container",
    "portcalls_dry_bulk",
    "portcalls_general_cargo",
    "portcalls_roro",
    "portcalls_tanker",
]
CAT_FEATURES = ["port"]
NUM_FEATURES = VESSEL_TYPES + [
    "portcalls_total",
    "day_of_week",
    "month",
    "import_roll7",  # mean daily import over the previous 7 days
    "portcalls_roll7",  # mean daily vessel calls over the previous 7 days
]
FEATURES = CAT_FEATURES + NUM_FEATURES

FEATURE_LABELS = {
    "port": "Port",
    "portcalls_container": "Container ships",
    "portcalls_dry_bulk": "Dry bulk carriers",
    "portcalls_general_cargo": "General cargo ships",
    "portcalls_roro": "Ro-Ro (vehicle) ships",
    "portcalls_tanker": "Tankers",
    "portcalls_total": "All vessel calls",
    "day_of_week": "Day of week",
    "month": "Month",
    "import_roll7": "Import, previous 7-day mean",
    "portcalls_roll7": "Calls, previous 7-day mean",
}


def load_params() -> dict:
    with open(ROOT / "params.yaml") as f:
        return yaml.safe_load(f)
