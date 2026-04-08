"""
FastAPI REST API for Water Potability Prediction.

Provides a /predict endpoint for programmatic access to the model.
"""

import os
import pickle

import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

app = FastAPI(
    title="Water Potability Prediction API",
    description="Predict whether water is safe to drink based on physicochemical properties.",
    version="1.0.0",
)


class WaterSample(BaseModel):
    ph: float = Field(..., ge=0, le=14, description="pH level (0-14)")
    Hardness: float = Field(..., ge=0, description="Hardness in mg/L")
    Solids: float = Field(..., ge=0, description="Total dissolved solids in mg/L")
    Chloramines: float = Field(..., ge=0, description="Chloramines in ppm")
    Sulfate: float = Field(..., ge=0, description="Sulfate in mg/L")
    Conductivity: float = Field(..., ge=0, description="Conductivity in μS/cm")
    Organic_carbon: float = Field(..., ge=0, description="Organic carbon in mg/L")
    Trihalomethanes: float = Field(..., ge=0, description="Trihalomethanes in μg/L")
    Turbidity: float = Field(..., ge=0, description="Turbidity in NTU")


class PredictionResponse(BaseModel):
    potable: bool
    confidence: float
    prediction_label: str


class BatchRequest(BaseModel):
    samples: list[WaterSample]


class BatchResponse(BaseModel):
    results: list[PredictionResponse]
    summary: dict


def get_model():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "models", "best_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(base, "models", "water_potability_random_forest.pkl")
    if not os.path.exists(model_path):
        raise HTTPException(status_code=500, detail="Model file not found.")
    with open(model_path, "rb") as f:
        return pickle.load(f)


_model = None


def load_model():
    global _model
    if _model is None:
        _model = get_model()
    return _model


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(sample: WaterSample):
    model = load_model()
    input_data = pd.DataFrame([sample.model_dump()])
    prediction = model.predict(input_data)[0]
    confidence = float(model.predict_proba(input_data)[0][1]) if hasattr(model, "predict_proba") else 0.5
    return PredictionResponse(
        potable=bool(prediction == 1),
        confidence=round(confidence, 4),
        prediction_label="Potable" if prediction == 1 else "Not Potable",
    )


@app.post("/predict/batch", response_model=BatchResponse)
def predict_batch(batch: BatchRequest):
    model = load_model()
    input_data = pd.DataFrame([s.model_dump() for s in batch.samples])
    predictions = model.predict(input_data)
    probabilities = model.predict_proba(input_data)[:, 1] if hasattr(model, "predict_proba") else [0.5] * len(predictions)

    results = []
    for pred, prob in zip(predictions, probabilities):
        results.append(PredictionResponse(
            potable=bool(pred == 1),
            confidence=round(float(prob), 4),
            prediction_label="Potable" if pred == 1 else "Not Potable",
        ))

    potable_count = int((predictions == 1).sum())
    return BatchResponse(
        results=results,
        summary={
            "total": len(predictions),
            "potable": potable_count,
            "not_potable": len(predictions) - potable_count,
        },
    )
