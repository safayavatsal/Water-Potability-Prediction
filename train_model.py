"""
Model training script for Water Potability Prediction.

Compares multiple classifiers (Random Forest, XGBoost, LightGBM, Gradient Boosting),
handles class imbalance with SMOTE, performs hyperparameter tuning with GridSearchCV,
and saves the best model along with evaluation metrics.
"""

import os
import json
import pickle
import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "water_potability.csv")
MODELS_DIR = os.path.join(os.path.dirname(__file__), "models")


def load_and_preprocess_data(path: str) -> tuple[pd.DataFrame, pd.Series]:
    """Load dataset and handle missing values + outliers."""
    df = pd.read_csv(path)

    # Fill missing values using group mean by Potability class
    for col in ["ph", "Sulfate", "Trihalomethanes"]:
        df[col] = df.groupby("Potability")[col].transform(
            lambda x: x.fillna(x.mean())
        )

    # IQR outlier clamping on Solids
    q1 = df["Solids"].quantile(0.25)
    q3 = df["Solids"].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    df["Solids"] = df["Solids"].clip(lower, upper)

    X = df.drop("Potability", axis=1)
    y = df["Potability"]
    return X, y


def get_models_and_params() -> dict:
    """Return dict of model name -> (estimator, param_grid)."""
    return {
        "RandomForest": (
            RandomForestClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [10, 20, None],
                "classifier__min_samples_split": [2, 5],
                "classifier__class_weight": ["balanced"],
            },
        ),
        "XGBoost": (
            XGBClassifier(
                random_state=42,
                eval_metric="logloss",
                use_label_encoder=False,
            ),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 6, 10],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__scale_pos_weight": [1.5],
            },
        ),
        "LightGBM": (
            LGBMClassifier(random_state=42, verbose=-1),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 6, 10],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
                "classifier__is_unbalance": [True],
            },
        ),
        "GradientBoosting": (
            GradientBoostingClassifier(random_state=42),
            {
                "classifier__n_estimators": [100, 200, 300],
                "classifier__max_depth": [3, 6, 10],
                "classifier__learning_rate": [0.01, 0.1, 0.2],
            },
        ),
    }


def train_and_evaluate():
    """Train all models, tune hyperparameters, select the best, and save it."""
    print("Loading and preprocessing data...")
    X, y = load_and_preprocess_data(DATA_PATH)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    models = get_models_and_params()
    results = {}

    for name, (estimator, param_grid) in models.items():
        print(f"\n{'='*60}")
        print(f"Training {name}...")
        print(f"{'='*60}")

        pipeline = ImbPipeline([
            ("smote", SMOTE(random_state=42)),
            ("scaler", StandardScaler()),
            ("classifier", estimator),
        ])

        grid = GridSearchCV(
            pipeline,
            param_grid,
            cv=cv,
            scoring="f1",
            n_jobs=-1,
            verbose=0,
        )
        grid.fit(X_train, y_train)

        y_pred = grid.predict(X_test)
        y_proba = grid.predict_proba(X_test)[:, 1]

        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_proba)

        results[name] = {
            "model": grid.best_estimator_,
            "best_params": grid.best_params_,
            "cv_best_f1": grid.best_score_,
            "test_accuracy": accuracy,
            "test_f1": f1,
            "test_precision": precision,
            "test_recall": recall,
            "test_roc_auc": roc_auc,
        }

        print(f"Best CV F1: {grid.best_score_:.4f}")
        print(f"Test Accuracy: {accuracy:.4f}")
        print(f"Test F1: {f1:.4f}")
        print(f"Test Precision: {precision:.4f}")
        print(f"Test Recall: {recall:.4f}")
        print(f"Test ROC-AUC: {roc_auc:.4f}")
        print(f"Best params: {grid.best_params_}")
        print(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
        print(f"Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

    # Select best model by test F1 score
    best_name = max(results, key=lambda k: results[k]["test_f1"])
    best = results[best_name]

    print(f"\n{'#'*60}")
    print(f"BEST MODEL: {best_name}")
    print(f"  Test F1: {best['test_f1']:.4f}")
    print(f"  Test ROC-AUC: {best['test_roc_auc']:.4f}")
    print(f"  Test Accuracy: {best['test_accuracy']:.4f}")
    print(f"{'#'*60}")

    # Save best model
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "best_model.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(best["model"], f)
    print(f"\nBest model saved to {model_path}")

    # Save all results as JSON (for later use in app)
    metrics = {}
    for name, res in results.items():
        metrics[name] = {
            k: v for k, v in res.items() if k != "model"
        }
        # Convert numpy types for JSON serialization
        for k, v in metrics[name].items():
            if isinstance(v, (np.floating, np.integer)):
                metrics[name][k] = float(v)

    metrics_path = os.path.join(MODELS_DIR, "model_comparison.json")
    with open(metrics_path, "w") as f:
        json.dump(
            {"best_model": best_name, "models": metrics},
            f,
            indent=2,
        )
    print(f"Model comparison saved to {metrics_path}")

    return best_name, best["model"]


if __name__ == "__main__":
    train_and_evaluate()
