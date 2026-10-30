.PHONY: install lint format test doctor dvc-auth repro experiments gate promote champion serve app

DAGSHUB_REPO ?= port-cargo-xai

install:            ## create .venv from uv.lock
	uv sync

lint:               ## code errors + style, changes nothing
	uv run ruff check .
	uv run ruff format --check .

format:             ## fix what lint reports
	uv run ruff check --fix .
	uv run ruff format .

test:
	uv run pytest

doctor:             ## check tools, secrets and the DagsHub connection
	uv run python -m src.doctor

dvc-auth:           ## store DagsHub login for DVC in .dvc/config.local (never committed)
	uv run dvc remote modify origin --local url "https://dagshub.com/$${DAGSHUB_OWNER}/$${DAGSHUB_REPO:-$(DAGSHUB_REPO)}.dvc"
	uv run dvc remote modify origin --local auth basic
	uv run dvc remote modify origin --local user "$${DAGSHUB_OWNER}"
	uv run dvc remote modify origin --local password "$${DAGSHUB_TOKEN}"

repro:              ## run the DVC pipeline: prepare -> train -> evaluate -> compare -> explain
	uv run dvc repro

experiments:        ## EBM vs standard models, logged to MLflow
	uv run python -m src.experiments

gate:               ## challenger must beat the @champion
	uv run python -m src.gate

promote:            ## register the model and make it @champion
	uv run python -m src.registry promote

champion:
	uv run python -m src.registry show

serve:              ## prediction API on port 8000 (/docs for the Swagger page)
	uv run uvicorn app.api:app --reload --host 0.0.0.0 --port 8000

app:                ## front end on port 8501
	uv run streamlit run app/streamlit_app.py --server.port 8501
