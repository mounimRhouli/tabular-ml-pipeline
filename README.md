# Tabular ML Pipeline

A small, reproducible machine-learning project for binary classification on
tabular data. It trains a scikit-learn pipeline on the built-in Wisconsin
Breast Cancer dataset, evaluates it, saves the fitted artifact, and exposes
predictions through FastAPI.

The dataset ships with scikit-learn, so no external data or credentials are
required.

## What is included

- deterministic, stratified train/test splitting
- feature scaling and logistic-regression training
- JSON evaluation metrics
- command-line and FastAPI predictions
- a DVC pipeline definition
- pytest coverage for preprocessing, training, prediction, and the API
- Docker and GitHub Actions configuration

## Project structure

```text
.
|-- .github/workflows/ci.yml
|-- model/                  # Generated artifacts (ignored by Git)
|-- serving/main.py         # FastAPI application
|-- src/
|   |-- data_preprocessing.py
|   |-- evaluate.py
|   |-- predict.py
|   `-- train.py
|-- tests/test_pipeline.py
|-- dvc.yaml
|-- params.yaml
|-- Dockerfile
`-- requirements.txt
```

## Local setup

Python 3.10 or newer is recommended.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train and evaluate

```powershell
python -m src.train
python -m src.evaluate
```

Training creates `model/model.joblib`. Evaluation creates
`model/metrics.json`. Both are generated files and are intentionally excluded
from Git.

To run the same workflow with DVC:

```powershell
dvc repro
dvc metrics show
```

## Make a command-line prediction

Provide exactly 30 numeric values in the dataset's documented feature order:

```powershell
python -m src.predict --features 14.1 20.0 92.0 600.0 0.1 0.12 0.1 0.05 0.18 0.06 0.4 1.2 3.0 40.0 0.006 0.02 0.03 0.01 0.02 0.003 16.0 27.0 105.0 800.0 0.14 0.3 0.35 0.15 0.25 0.08
```

## Run the API

Train the model first, then start the server:

```powershell
uvicorn serving.main:app --host 0.0.0.0 --port 8000
```

Interactive API documentation is available at `http://localhost:8000/docs`.
The `/metadata` endpoint returns the expected feature names and order.

With Docker:

```powershell
docker compose up --build
```

## Tests

```powershell
pytest
```

## Configuration

Edit `params.yaml` to change the test split, random seed, or logistic-regression
hyperparameters. Re-run training and evaluation after changing parameters.
