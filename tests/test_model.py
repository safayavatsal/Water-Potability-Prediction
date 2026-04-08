"""Tests for model loading, prediction, and feature engineering."""

import os
import pickle

import numpy as np
import pandas as pd
import pytest

SAMPLE_INPUT = pd.DataFrame({
    "ph": [7.0],
    "Hardness": [150.0],
    "Solids": [500.0],
    "Chloramines": [2.0],
    "Sulfate": [100.0],
    "Conductivity": [400.0],
    "Organic_carbon": [1.0],
    "Trihalomethanes": [50.0],
    "Turbidity": [1.0],
})

BASE_DIR = os.path.join(os.path.dirname(__file__), "..")


@pytest.fixture
def model():
    model_path = os.path.join(BASE_DIR, "models", "best_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(BASE_DIR, "models", "water_potability_random_forest.pkl")
    with open(model_path, "rb") as f:
        return pickle.load(f)


@pytest.fixture
def dataset():
    return pd.read_csv(os.path.join(BASE_DIR, "data", "water_potability.csv"))


class TestModelLoading:
    def test_model_file_exists(self):
        best = os.path.join(BASE_DIR, "models", "best_model.pkl")
        legacy = os.path.join(BASE_DIR, "models", "water_potability_random_forest.pkl")
        assert os.path.exists(best) or os.path.exists(legacy)

    def test_model_has_predict(self, model):
        assert hasattr(model, "predict")

    def test_model_has_predict_proba(self, model):
        assert hasattr(model, "predict_proba")


class TestPrediction:
    def test_predict_returns_binary(self, model):
        pred = model.predict(SAMPLE_INPUT)
        assert pred[0] in (0, 1)

    def test_predict_proba_returns_two_classes(self, model):
        proba = model.predict_proba(SAMPLE_INPUT)
        assert proba.shape == (1, 2)

    def test_predict_proba_sums_to_one(self, model):
        proba = model.predict_proba(SAMPLE_INPUT)
        assert abs(proba[0].sum() - 1.0) < 1e-6

    def test_predict_batch(self, model):
        batch = pd.concat([SAMPLE_INPUT] * 10, ignore_index=True)
        preds = model.predict(batch)
        assert len(preds) == 10
        assert all(p in (0, 1) for p in preds)

    def test_predict_edge_case_zeros(self, model):
        zeros = pd.DataFrame({col: [0.0] for col in SAMPLE_INPUT.columns})
        pred = model.predict(zeros)
        assert pred[0] in (0, 1)

    def test_predict_edge_case_max_values(self, model):
        maxvals = pd.DataFrame({
            "ph": [14.0], "Hardness": [500.0], "Solids": [60000.0],
            "Chloramines": [15.0], "Sulfate": [500.0], "Conductivity": [800.0],
            "Organic_carbon": [25.0], "Trihalomethanes": [120.0], "Turbidity": [7.0],
        })
        pred = model.predict(maxvals)
        assert pred[0] in (0, 1)


class TestDataset:
    def test_dataset_exists(self, dataset):
        assert len(dataset) > 0

    def test_dataset_has_required_columns(self, dataset):
        required = ["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                     "Conductivity", "Organic_carbon", "Trihalomethanes",
                     "Turbidity", "Potability"]
        for col in required:
            assert col in dataset.columns

    def test_potability_is_binary(self, dataset):
        assert set(dataset["Potability"].dropna().unique()) == {0, 1}

    def test_dataset_row_count(self, dataset):
        assert len(dataset) > 3000


class TestFeatureEngineering:
    def test_feature_engineer_transform(self):
        from features import WaterFeatureEngineer
        fe = WaterFeatureEngineer()
        result = fe.transform(SAMPLE_INPUT)
        assert "ph_safe" in result.columns
        assert "who_violations_count" in result.columns
        assert "ph_x_turbidity" in result.columns
        assert "solids_per_conductivity" in result.columns

    def test_feature_engineer_who_flags(self):
        from features import WaterFeatureEngineer
        fe = WaterFeatureEngineer()
        # pH=7 is within safe range
        result = fe.transform(SAMPLE_INPUT)
        assert result["ph_safe"].iloc[0] == 1

    def test_feature_engineer_who_violation(self):
        from features import WaterFeatureEngineer
        fe = WaterFeatureEngineer()
        bad = SAMPLE_INPUT.copy()
        bad["ph"] = 3.0  # below WHO min of 6.5
        result = fe.transform(bad)
        assert result["ph_safe"].iloc[0] == 0
