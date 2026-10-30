"""Quality gate: the new model (challenger) must not be worse than the current @champion.

Usage:  python -m src.gate        exit 0 = pass, 1 = fail; writes gate_report.md
"""

import json
import sys

from mlflow import MlflowClient
from mlflow.exceptions import MlflowException

from . import tracking
from .config import METRICS, ROOT, load_params

REPORT_METRICS = ["test_mae", "test_rmse", "test_r2", "test_wape", "mean_prediction"]


def champion_metrics(name: str) -> dict | None:
    client = MlflowClient()
    try:
        version = client.get_model_version_by_alias(name, "champion")
    except MlflowException:
        return None
    run = client.get_run(version.run_id)
    return {"version": version.version, **run.data.metrics}


def decide(challenger: float, champion: float | None, tolerance: float) -> bool:
    """Lower is better (MAE). No champion yet -> the first model passes."""
    return champion is None or challenger <= champion * (1 + tolerance)


def main() -> int:
    p = load_params()
    metric, tol, name = p["gate"]["metric"], p["gate"]["tolerance"], p["model"]["name"]
    tracking.setup("port-cargo-ebm")
    challenger = json.loads((METRICS / "eval_metrics.json").read_text())
    champ = champion_metrics(name)
    passed = decide(challenger[metric], champ.get(metric) if champ else None, tol)

    def fmt(d, k):
        return "none yet" if not d or k not in d else f"{d[k]:,.3f}"

    lines = [
        f"## Quality gate: {'PASS' if passed else 'FAIL'}",
        "",
        f"Rule: challenger `{metric}` <= champion `{metric}` x (1 + {tol}). Lower is better.",
        f"Champion: {'version ' + str(champ['version']) if champ else 'none yet'}",
        "",
        "| Metric | Champion | Challenger |",
        "|---|---:|---:|",
        *[f"| {k} | {fmt(champ, k)} | {fmt(challenger, k)} |" for k in REPORT_METRICS],
    ]
    (ROOT / "gate_report.md").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
