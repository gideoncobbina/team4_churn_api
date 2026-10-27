from pathlib import Path
import json

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field


APP_DIR = Path(__file__).resolve().parents[1]

MODEL_PATH = (
    APP_DIR
    / "ml_platform"
    / "training"
    / "churn"
    / "models"
    / "churn_pipeline.joblib"
)

FEATURES_PATH = (
    APP_DIR
    / "ml_platform"
    / "training"
    / "churn"
    / "models"
    / "feature_names.json"
)

def norm(name: str) -> str:
    return " ".join(name.split())

app = FastAPI(
    title="Ishango Telecom Churn API",
    version="1.0.0",
    description="Week 1 production API for telecom churn prediction.",
)


class PredictionRequest(BaseModel):
    call_failure: int = Field(ge=0)
    complains: int = Field(ge=0, le=1)
    subscription_length: int = Field(ge=0)
    charge_amount: int = Field(ge=0)
    seconds_of_use: int = Field(ge=0)
    frequency_of_use: int = Field(ge=0)
    frequency_of_sms: int = Field(ge=0)
    distinct_called_numbers: int = Field(ge=0)
    age_group: int = Field(ge=1)
    tariff_plan: int = Field(ge=1)
    status: int = Field(ge=1)
    age: int = Field(ge=0)
    customer_value: float = Field(ge=0)


class PredictionResponse(BaseModel):
    prediction: int
    churn_probability: float
    model_version: str


if not MODEL_PATH.exists():
    raise RuntimeError(
        f"Model artifact not found at {MODEL_PATH}. "
        "Run the Week 1 anchor training step first."
    )

if not FEATURES_PATH.exists():
    raise RuntimeError(
        f"Feature metadata not found at {FEATURES_PATH}."
    )


model = joblib.load(MODEL_PATH)
feature_names = json.loads(FEATURES_PATH.read_text())

@app.get("/", include_in_schema=False)
def root() -> dict[str, str]:
    return {
        "service": "telecom-churn-api",
        "docs": "/docs",
        "health": "/health",
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
    return Response(status_code=204)


@app.get("/health")
def health() -> dict[str, str]:
    return {
        "status": "ok",
        "service": "telecom-churn-api",
    }


@app.post(
    "/predictions",
    response_model=PredictionResponse,
)
def predict(
    payload: PredictionRequest,
) -> PredictionResponse:
    api_values = {
        norm("Call Failure"): payload.call_failure,
        norm("Complains"): payload.complains,
        norm("Subscription Length"): payload.subscription_length,
        norm("Charge Amount"): payload.charge_amount,
        norm("Seconds of Use"): payload.seconds_of_use,
        norm("Frequency of use"): payload.frequency_of_use,
        norm("Frequency of SMS"): payload.frequency_of_sms,
        norm("Distinct Called Numbers"): payload.distinct_called_numbers,
        norm("Age Group"): payload.age_group,
        norm("Tariff Plan"): payload.tariff_plan,
        norm("Status"): payload.status,
        norm("Age"): payload.age,
        norm("Customer Value"): payload.customer_value,
    }

    try:
        row = pd.DataFrame(
            [[api_values[norm(name)] for name in feature_names]],
            columns=feature_names,
        )

        prediction = int(model.predict(row)[0])
        probability = float(model.predict_proba(row)[0][1])

        return PredictionResponse(
            prediction=prediction,
            churn_probability=round(probability, 4),
            model_version="week1-v1",
        )

    except Exception as exc:
        logger.exception("Prediction failed")
        raise HTTPException(
            status_code=500,
            detail="Prediction failed.",
        ) from exc