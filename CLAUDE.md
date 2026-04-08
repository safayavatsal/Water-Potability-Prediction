# Water Potability Prediction

## Project Overview

A machine learning web application that predicts whether water is safe for human consumption based on 9 physicochemical properties. Uses the best model from a comparative evaluation (Random Forest, XGBoost, LightGBM, Gradient Boosting) with SMOTE oversampling and hyperparameter tuning, deployed via Streamlit with a FastAPI REST API backend.

## Tech Stack

- **Language:** Python 3.11+
- **Web Framework:** Streamlit (frontend), FastAPI (REST API)
- **ML Libraries:** scikit-learn, XGBoost, LightGBM, imbalanced-learn (SMOTE)
- **Explainability:** SHAP
- **Visualization:** Plotly, matplotlib
- **Data Processing:** pandas, numpy
- **Testing:** pytest, httpx
- **CI/CD:** GitHub Actions
- **Containerization:** Docker, Docker Compose

## Project Structure

```
├── app.py                  # Multi-page Streamlit web app (Predict, Batch, Explore, Performance, About)
├── api.py                  # FastAPI REST API (/predict, /predict/batch, /health)
├── train_model.py          # Model training: multi-model comparison + GridSearchCV + SMOTE
├── features.py             # Feature engineering: WHO threshold flags + interaction features
├── explainability.py       # SHAP-based model explanations (global + per-prediction)
├── data/
│   └── water_potability.csv
├── models/
│   ├── best_model.pkl      # Best model pipeline (features + SMOTE + scaler + classifier)
│   └── model_comparison.json
├── tests/
│   ├── test_model.py       # Model, prediction, dataset, and feature engineering tests
│   └── test_api.py         # FastAPI endpoint tests
├── .github/workflows/
│   └── ci.yml              # GitHub Actions: pytest + ruff linting
├── Dockerfile              # Production container image
├── docker-compose.yml      # Runs Streamlit (8501) + FastAPI (8000)
├── .devcontainer/
│   └── devcontainer.json
├── requirements.txt
├── LICENSE                 # MIT License
└── README.md
```

## Running

```bash
# Install dependencies
pip install -r requirements.txt

# Run Streamlit app
streamlit run app.py

# Run FastAPI
uvicorn api:app --reload

# Run tests
pytest tests/ -v

# Retrain models
python train_model.py

# Docker
docker compose up
```

## Model Pipeline

Pipeline: `WaterFeatureEngineer → SMOTE → StandardScaler → Classifier`

- **Feature Engineering:** WHO threshold binary flags, violation count, interaction features (ph×turbidity, chloramines/organic_carbon, solids/conductivity)
- **Class Imbalance:** SMOTE oversampling (60/40 → balanced)
- **Hyperparameter Tuning:** GridSearchCV with 5-fold stratified CV, scored by F1
- **Model Selection:** Best model by test F1 across RF, XGBoost, LightGBM, GradientBoosting

## API Endpoints

- `GET /health` — readiness check
- `POST /predict` — single sample prediction (returns potable, confidence, label)
- `POST /predict/batch` — batch prediction
- `GET /docs` — Swagger UI (auto-generated)

## Dataset Features (Input Order)

1. `ph` (0–14), 2. `Hardness` (mg/L), 3. `Solids` (mg/L), 4. `Chloramines` (ppm),
5. `Sulfate` (mg/L), 6. `Conductivity` (μS/cm), 7. `Organic_carbon` (mg/L),
8. `Trihalomethanes` (μg/L), 9. `Turbidity` (NTU)
