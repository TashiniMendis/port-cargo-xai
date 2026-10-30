"""Health check of the environment.  Usage:  make doctor"""

import os
import platform
import shutil
import subprocess
import sys
from importlib.metadata import version

OK, BAD = "OK  ", "FAIL"


def check(label, ok, detail=""):
    print(f"[{OK if ok else BAD}] {label} {detail}")
    return ok


def main():
    results = [
        check("Python", platform.python_version().startswith("3.12"), platform.python_version())
    ]
    for pkg in ["interpret-core", "scikit-learn", "mlflow", "dvc", "fastapi", "streamlit"]:
        try:
            results.append(check(pkg, True, version(pkg)))
        except Exception:
            results.append(check(pkg, False, "not installed - run `uv sync`"))
    uv = shutil.which("uv")
    results.append(check("uv", uv is not None, uv or "not found - pip install --user uv"))
    for var in ["DAGSHUB_OWNER", "DAGSHUB_TOKEN"]:
        results.append(
            check(
                var,
                bool(os.getenv(var)),
                "set" if os.getenv(var) else "not set (local MLflow store will be used)",
            )
        )
    if os.getenv("DAGSHUB_OWNER") and os.getenv("DAGSHUB_TOKEN"):
        from mlflow import MlflowClient

        from . import tracking

        uri = tracking.setup("port-cargo-ebm")
        try:
            MlflowClient().search_experiments(max_results=1)
            results.append(check("DagsHub MLflow", True, uri))
        except Exception as e:
            results.append(check("DagsHub MLflow", False, str(e)[:120]))
    r = subprocess.run(
        [sys.executable, "-m", "dvc", "remote", "list"], capture_output=True, text=True
    )
    results.append(check("DVC remote", "origin" in r.stdout, r.stdout.strip() or "none"))
    print("\nAll checks passed." if all(results) else "\nSome checks failed (see above).")


if __name__ == "__main__":
    main()
