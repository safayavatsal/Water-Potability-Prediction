"""
SHAP-based model explainability for Water Potability Prediction.

Generates global feature importance and per-prediction explanations.
"""

import os
import pickle

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import shap


def load_model():
    """Load the best trained model pipeline."""
    base = os.path.dirname(__file__)
    path = os.path.join(base, "models", "best_model.pkl")
    if not os.path.exists(path):
        path = os.path.join(base, "models", "water_potability_random_forest.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


def load_data():
    """Load and preprocess the dataset (same as training)."""
    base = os.path.dirname(__file__)
    df = pd.read_csv(os.path.join(base, "data", "water_potability.csv"))
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        df[col] = df.groupby("Potability")[col].transform(
            lambda x: x.fillna(x.mean())
        )
    q1 = df["Solids"].quantile(0.25)
    q3 = df["Solids"].quantile(0.75)
    iqr = q3 - q1
    df["Solids"] = df["Solids"].clip(q1 - 1.5 * iqr, q3 + 1.5 * iqr)
    X = df.drop("Potability", axis=1)
    return X


def get_shap_explainer(model, X_background):
    """Create a SHAP explainer for the model pipeline.

    Handles both Pipeline objects and plain estimators.
    Uses a subsample of data as background for speed.
    """
    background = shap.sample(X_background, min(100, len(X_background)))

    # If it's a pipeline, we need to wrap the predict function
    if hasattr(model, "predict_proba"):
        explainer = shap.Explainer(model.predict_proba, background)
    else:
        explainer = shap.Explainer(model.predict, background)
    return explainer


def compute_shap_values(model, X_background, input_data):
    """Compute SHAP values for given input data."""
    explainer = get_shap_explainer(model, X_background)
    shap_values = explainer(input_data)
    return shap_values


def plot_global_feature_importance(model, X_background):
    """Generate and return a global SHAP feature importance bar plot."""
    background = shap.sample(X_background, min(200, len(X_background)))
    explainer = get_shap_explainer(model, background)
    shap_values = explainer(background)

    fig, ax = plt.subplots(figsize=(8, 5))
    # For multi-output, take potable class (index 1)
    if len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    shap.plots.bar(sv, show=False, ax=ax)
    ax.set_title("Feature Importance (SHAP)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_waterfall(shap_values, index=0):
    """Generate a SHAP waterfall plot for a single prediction."""
    fig = plt.figure(figsize=(8, 5))
    # For multi-output, take potable class (index 1)
    if len(shap_values.shape) == 3:
        sv = shap_values[index, :, 1]
    else:
        sv = shap_values[index]
    shap.plots.waterfall(sv, show=False)
    plt.title("Prediction Explanation (SHAP)", fontsize=14)
    plt.tight_layout()
    return fig


def plot_beeswarm(model, X_background):
    """Generate a SHAP beeswarm (summary) plot."""
    background = shap.sample(X_background, min(200, len(X_background)))
    explainer = get_shap_explainer(model, background)
    shap_values = explainer(background)

    fig = plt.figure(figsize=(8, 6))
    if len(shap_values.shape) == 3:
        sv = shap_values[:, :, 1]
    else:
        sv = shap_values
    shap.plots.beeswarm(sv, show=False)
    plt.title("Feature Impact Distribution (SHAP)", fontsize=14)
    plt.tight_layout()
    return fig
