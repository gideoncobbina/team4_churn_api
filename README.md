# Telecom Customer Churn API

A production-style machine learning service that predicts whether a telecom customer is likely to churn. The project takes a simple, reproducible scikit-learn model through the full delivery path: **train → serve → containerize → test in CI → release.**

> **Status:** Week 1 teaching project (Ishango.ai Production AI Engineering Programme). The API is not hardened for sensitive public use.

---

## Table of Contents

1. [Overview](#overview)
2. [Architecture](#architecture)
3. [Repository Structure](#repository-structure)
4. [Prerequisites](#prerequisites)
5. [Getting Started](#getting-started)
6. [Train the Model](#train-the-model)
7. [Run the API Locally](#run-the-api-locally)
8. [API Reference](#api-reference)
9. [Run with Docker](#run-with-docker)
10. [Quality Checks and Tests](#quality-checks-and-tests)
11. [Team Workflow and CI](#team-workflow-and-ci)
12. [Creating a Release](#creating-a-release)
13. [Troubleshooting](#troubleshooting)
14. [Optional: Cloud Deployment](#optional-cloud-deployment)
15. [Contributing](#contributing)

---

## Overview

| Item | Detail |
|---|---|
| **Task** | Binary classification: will a customer churn (1) or stay (0)? |
| **Data** | UCI Machine Learning Repository, dataset ID 563, fetched with `ucimlrepo` |
| **Model** | Scikit-learn `Pipeline`: median imputer → standard scaler → logistic regression (`class_weight="balanced"`) |
| **Service** | FastAPI app with Pydantic request validation, served by Uvicorn |
| **Packaging** | Docker image built from the repository root |
| **Automation** | GitHub Actions runs lint and tests on every pull request |

The full preprocessing-plus-model pipeline is saved as a single artifact, so the same transformations used in training are applied at prediction time. This prevents training–serving mismatch.

## Architecture

```
UCI dataset
    │
    ▼
train.py ──► churn_pipeline.joblib + feature_names.json
                          │
                          ▼
              api/main.py  (FastAPI app, validates input, loads artifact once at startup)
                          │
                          ▼
         Dockerfile + entrypoint.sh  (api.main:app on port 8080)
                          │
                          ▼
        GitHub Actions CI ──► reviewed PR ──► tagged release
```

**The deployment contract is `api.main:app`:** the object `app` inside `api/main.py`. Uvicorn, Docker, and any hosting platform rely on that exact path, so do not rename the `api/` folder, the file, or the object.

## Repository Structure

```
.
├── api/
│   ├── __init__.py
│   └── main.py                  # FastAPI service
├── ml_platform/
│   └── training/
│       └── churn/
│           ├── train.py         # Training and evaluation script
│           └── models/
│               ├── churn_pipeline.joblib   # Saved fitted pipeline
│               └── feature_names.json      # Expected input columns, in order
├── production_app/
│   └── tests/
│       └── test_api.py          # API tests
├── .github/workflows/checks.yml # CI workflow
├── Dockerfile
├── entrypoint.sh
├── .dockerignore
├── .gitignore
├── .gitattributes
├── pyproject.toml
├── uv.lock
└── README.md
```

## Prerequisites

- **Python 3.12**
- **[uv](https://docs.astral.sh/uv/)** (the project pins version `0.11.2` in CI)
- **Git**
- **Docker** with a running engine (verify with `docker --version` and `docker info`)

If Docker is unavailable in your environment, use a teammate's machine or the course Codespace.

## Getting Started

```bash
git clone https://github.com/YOUR-TEAM-OWNER/YOUR-TEAM-REPO.git
cd YOUR-TEAM-REPO
uv sync --locked
```

`--locked` installs the exact dependency versions recorded in `uv.lock`, so every team member has an identical environment.

## Train the Model

Training produces the two artifacts the API needs. It fetches the dataset, splits it, fits the pipeline, evaluates it, and saves the results.

```bash
mkdir -p ml_platform/training/churn/models
uv run python ml_platform/training/churn/train.py
```

What the script does:

1. **Fetch data** with `fetch_ucirepo(id=563)`; `X` is the 13 input features and `y` is the churn label.
2. **Split** with `train_test_split(test_size=0.2, random_state=42, stratify=y)`. The split is reproducible and keeps the churn ratio equal in both sets.
3. **Fit** the pipeline on training data only.
4. **Evaluate** on held-out data using `classification_report` and `roc_auc_score`.
5. **Save** `churn_pipeline.joblib` and `feature_names.json` to `ml_platform/training/churn/models/`.

**Evaluating the model:** do not rely on accuracy alone. Churn data is imbalanced, so a model that predicts "no churn" for everyone can look accurate and be useless. Review **recall for the churn class** and ROC AUC. A false negative is a customer who was about to leave and received no retention offer.

Verify the artifacts exist:

```bash
ls -lh ml_platform/training/churn/models/
# churn_pipeline.joblib
# feature_names.json
```

## Run the API Locally

```bash
uv run uvicorn api.main:app --host 0.0.0.0 --port 8080
```

When you see `Application startup complete`, the service is ready.

- Health check: <http://localhost:8080/health>
- Interactive docs (Swagger UI): <http://localhost:8080/docs>
- OpenAPI schema: <http://localhost:8080/openapi.json>

The service **refuses to start** if the model artifact is missing. This is deliberate: a service that starts but cannot predict is worse than one that fails loudly.

## API Reference

### `GET /health`

Lightweight liveness check. Does not run the model.

```bash
curl http://localhost:8080/health
```

```json
{"status": "ok", "service": "telecom-churn-api"}
```

### `POST /predictions`

Returns a churn prediction for one customer.

**Request fields** (all required):

| Field | Type | Constraint |
|---|---|---|
| `call_failure` | int | ≥ 0 |
| `complains` | int | 0 or 1 |
| `subscription_length` | int | per schema |
| `charge_amount` | int | per schema |
| `seconds_of_use` | int | per schema |
| `frequency_of_use` | int | per schema |
| `frequency_of_sms` | int | per schema |
| `distinct_called_numbers` | int | per schema |
| `age_group` | int | per schema |
| `tariff_plan` | int | per schema |
| `status` | int | per schema |
| `age` | int | per schema |
| `customer_value` | float | ≥ 0 |

Exact constraints are defined by the `PredictionRequest` class in `api/main.py`. The API accepts friendly snake_case names and maps them internally to the original training column names (for example `call_failure` → `"Call Failure"`), ordered according to `feature_names.json`.

**Example request:**

```bash
curl -X POST http://localhost:8080/predictions \
  -H "Content-Type: application/json" \
  -d '{
    "call_failure": 8,
    "complains": 1,
    "subscription_length": 20,
    "charge_amount": 3,
    "seconds_of_use": 2500,
    "frequency_of_use": 55,
    "frequency_of_sms": 12,
    "distinct_called_numbers": 25,
    "age_group": 3,
    "tariff_plan": 1,
    "status": 1,
    "age": 42,
    "customer_value": 1200.5
  }'
```

**Response:**

```json
{
  "prediction": 0,
  "churn_probability": 0.2741,
  "model_version": "week1-v1"
}
```

The values above are illustrative. Actual probabilities depend on your trained artifact.

**Error responses:**

| Status | Meaning |
|---|---|
| `422` | Request failed validation (missing field, wrong type, out-of-range value). The model never runs. |
| `500` | Internal prediction failure. A generic message is returned; details are in the server logs. |

## Run with Docker

All commands run from the **repository root**, which is the build context.

**1. Confirm the build context**

```bash
ls
# api/  ml_platform/  pyproject.toml  uv.lock  Dockerfile  entrypoint.sh
ls -lh ml_platform/training/churn/models/
cat .dockerignore   # must NOT exclude the model artifacts
```

**2. Build the image**

```bash
docker build -t team-churn:v1 .
docker images team-churn
```

**3. Run the container**

```bash
docker run --rm --name ishango-week1-api -p 8080:8080 team-churn:v1
```

`-p HOST:CONTAINER` maps a port on your machine to a port inside the container. For example, `-p 9999:8080` serves the same app at `localhost:9999`.

**4. Test it** from a second terminal using the same `curl` commands as above.

**5. Inspect the running container**

```bash
docker ps
docker logs ishango-week1-api
docker inspect ishango-week1-api
docker exec ishango-week1-api ls -lh /app/ml_platform/training/churn/models/
```

**6. Stop it**

```bash
docker stop ishango-week1-api
```

**Key behaviors to understand**

- **Images are immutable snapshots.** Editing code does not change an image that is already built. Any code change requires a new build (for example `team-churn:v2`).
- The model is **packaged inside the image** at `/app/ml_platform/training/churn/models/`.
- `entrypoint.sh` starts Uvicorn on `${PORT:-8080}`, so hosting platforms can supply their own port.
- Keep `entrypoint.sh` saved with **LF line endings**. CRLF endings can cause confusing startup failures on Linux.

## Quality Checks and Tests

Run these from the repository root before every commit:

```bash
uv sync --locked
uv run --locked ruff check .
uv run ruff format --check .
uv run --locked pytest
```

| Command | Purpose |
|---|---|
| `ruff check` | Lint: finds likely mistakes and style problems |
| `ruff format --check` | Confirms consistent formatting |
| `pytest` | Runs the automated tests |

The tests use FastAPI's `TestClient` and cover at least the health endpoint and rejection of invalid input (HTTP 422). Because the API loads the model at import time, **the tests require the saved model artifact.**

Recommended additions: a valid-prediction test and further failure-case tests.

## Team Workflow and CI

### Branching and pull requests

1. Work on an issue branch, never on `main`.
   ```bash
   git branch --show-current
   git status
   ```
2. Review your changes before committing.
   ```bash
   git diff --name-only origin/main...HEAD
   git diff
   ```
3. Stage specific files (avoid committing `.venv`, secrets, or raw customer data).
   ```bash
   git add api ml_platform production_app pyproject.toml uv.lock Dockerfile entrypoint.sh .dockerignore .gitignore .gitattributes
   git commit -m "feat: package team churn service"
   git push -u origin HEAD
   ```
4. Open a pull request into `main`. Describe the change, how you tested it, and any model or API-contract implications. Request a review from another team member.

Each member should author at least one meaningful PR and review at least one other.

### Continuous integration

`.github/workflows/checks.yml` runs on pull requests to `main` and on pushes to `main`:

```yaml
name: Team quality checks
on:
  pull_request:
    branches: [main]
  push:
    branches: [main]
permissions:
  contents: read
jobs:
  checks:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v6
        with:
          version: "0.11.2"
          python-version: "3.14"
      - run: uv sync --locked
      - run: uv run --locked ruff check .
      - run: uv run --locked pytest
```

**CI only checks code. It does not deploy.** Merging into `main` updates the repository but does not restart or redeploy any running service.

### When CI fails

Find the **first** meaningful error and classify it:

| Failure type | Typical cause |
|---|---|
| Lockfile mismatch | `pyproject.toml` and `uv.lock` are out of sync |
| Missing artifact | Model files were not committed |
| Lint failure | Ruff reported a code or style problem |
| Test failure | A test expectation was not met |
| Runner/network failure | Temporary infrastructure problem |

Reproduce the failing command locally, fix the cause, and push to the same PR. Do not disable checks to get a green result.

### Model artifacts in Git

For this teaching project, commit the trusted `churn_pipeline.joblib` and `feature_names.json` so CI and teammates use the identical artifact without re-downloading data. **Never commit secrets or raw customer records.** In larger projects, move artifacts to dedicated artifact storage.

## Creating a Release

**1. Verify reproducibility from a fresh clone** (performed by a teammate, using only this README):

```bash
git clone https://github.com/YOUR-TEAM-OWNER/YOUR-TEAM-REPO.git
cd YOUR-TEAM-REPO
uv sync --locked
uv run --locked pytest
```

**2. Build the release image and record the commit:**

```bash
git rev-parse HEAD
docker build -t team-churn:week1-v1 .
```

Record the commit SHA alongside the image tag. You may also tag the image with the SHA for traceability.

**3. Run and verify the release locally:**

```bash
docker run --rm --name team-churn-release -p 8080:8080 team-churn:week1-v1
curl http://localhost:8080/health
curl http://localhost:8080/openapi.json
```

Then repeat the valid and invalid prediction requests. Confirm a prediction, probability, and version for the valid one and HTTP `422` for the invalid one.

**4. Capture evidence:**

```bash
docker logs team-churn-release
docker inspect team-churn-release
git rev-parse HEAD
```

Do not log full customer records or credentials.

**5. Tag the release:**

```bash
git switch main
git pull --ff-only
git tag week1-v1
git push origin week1-v1
```

**Submission checklist**

- [ ] Repository URL and release tag/commit
- [ ] Evidence of passing tests
- [ ] Evidence of container predictions (valid request and rejected request)
- [ ] Contribution table linking each member's PR and review
- [ ] Facilitator granted access (if the repository is private)

## Troubleshooting

| Symptom | Where to look |
|---|---|
| `uvicorn` cannot import `api.main` | Wrong working directory, missing `api/__init__.py`, or renamed file/object |
| Service refuses to start | Model artifact missing from `ml_platform/training/churn/models/` |
| `/health` works, `/predictions` returns 500 | Schema-to-column mapping, artifact compatibility, or inference logic |
| `422` response | Input failed validation (the system working as designed) |
| Predictions differ from the notebook | Training–serving mismatch: column order, names, or preprocessing |
| Works for a teammate, not for you | Your copy is missing the artifact or has a path problem |
| CI green but `docker build` fails | Dockerfile, build context, or model packaging (CI does not test Docker) |
| Container exits immediately | Entrypoint, import path, missing artifact, or CRLF line endings in `entrypoint.sh` |
| Container running but unreachable | Port mapping (`-p host:container`) or bind address (must be `0.0.0.0`) |
| `docker run` fails before the app starts | Image tag does not exist locally |
| Cloud image push denied | Registry authentication and permissions |

General rule: **identify which layer is failing before changing code.**

## Optional: Cloud Deployment

Cloud deployment is optional and earns no advantage over an equivalent local demonstration. If your team chooses to deploy:

- A **registry** stores versioned images; a **runtime** runs them. Use any provider that can run a Docker container or a Python ASGI service (for example Google Cloud Run).
- Check the provider's current pricing, account requirements, and instructions before provisioning.
- Deploy the image built from the reviewed commit. Bind to `0.0.0.0`, supply `PORT`, and configure `/health`.
- Use least-privilege access. Never commit credentials. Prefer OIDC/workload identity over long-lived CI keys where supported.
- Record the image tag, commit SHA, service URL, resource owner, and cleanup steps.
- Verify the hosted service:
  ```bash
  export SERVICE_URL="https://YOUR-TEAM-SERVICE-URL"
  curl "$SERVICE_URL/health"
  curl "$SERVICE_URL/openapi.json"
  ```
- Add continuous delivery only after the manual route is understood: **review → tests → build → publish image → deploy → verify.**

## Contributing

1. Pick or create an issue and branch from `main`.
2. Keep pull requests small and focused; explain any change outside your assigned issue.
3. Run all quality checks locally before pushing.
4. Request review from a teammate. Merge only after CI passes and a review is complete.
5. Rotate responsibilities (training, API, tests, packaging) so every member practices each area.

---

*Maintained by the project team. Replace the placeholder repository URLs above with your team's repository.*