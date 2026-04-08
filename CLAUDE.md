# Water Potability Prediction

## Project Overview

A machine learning web application that predicts whether water is safe for human consumption based on 9 physicochemical properties. Uses a trained Random Forest Classifier deployed via Streamlit.

## Tech Stack

- **Language:** Python 3.11
- **Web Framework:** Streamlit
- **ML Library:** scikit-learn (RandomForestClassifier)
- **Data Processing:** pandas, numpy
- **Visualization (notebook):** matplotlib, seaborn
- **Model Serialization:** pickle

## Project Structure

```
├── app.py                  # Streamlit web app — main entry point
├── data/
│   └── water_potability.csv  # Dataset (3,276 samples, 9 features + target)
├── models/
│   └── water_potability_random_forest.pkl  # Trained RF model (~5.2 MB)
├── notebook/
│   └── Water Potbaility_ML Project.ipynb   # EDA, preprocessing, training notebook
├── .devcontainer/
│   └── devcontainer.json   # GitHub Codespaces / VS Code dev container config
├── requirements.txt        # Python dependencies (no version pinning)
├── LICENSE                 # MIT License (Dhruv Jaradi, 2025)
└── README.md               # Project documentation
```

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```

The app runs on port 8501 by default.

## Model Details

- **Algorithm:** Random Forest Classifier (default hyperparameters, `random_state=42`)
- **Train/Test Split:** 80/20
- **Accuracy:** ~83%
- **Class Performance:**
  - Non-potable (0): Precision=0.81, Recall=0.94, F1=0.87
  - Potable (1): Precision=0.87, Recall=0.64, F1=0.73

## Dataset Features (Input Order)

The model expects features in this exact order:

1. `ph` — Acidity/alkalinity (0–14)
2. `Hardness` — Mineral concentration (mg/L)
3. `Solids` — Total dissolved solids (mg/L)
4. `Chloramines` — Disinfectant level (ppm)
5. `Sulfate` — Chemical compound (mg/L)
6. `Conductivity` — Electrical conductivity (μS/cm)
7. `Organic_carbon` — Organic contaminants (mg/L)
8. `Trihalomethanes` — Disinfection byproducts (μg/L)
9. `Turbidity` — Water clarity (NTU)

**Important:** Feature names in prediction DataFrames must match training data exactly (e.g., `Organic_carbon`, not `Organic Carbon`).

## Data Preprocessing (in notebook)

- Missing values in `ph`, `Sulfate`, `Trihalomethanes` filled using grouped mean by Potability class
- Outliers in `Solids` handled via IQR clamping (1.5×IQR threshold)

## Development Notes

- The `.devcontainer/` config auto-installs dependencies and starts the Streamlit app on port 8501
- CORS and XSRF protection are disabled in devcontainer for local development convenience
- The dataset has class imbalance: ~60% non-potable, ~40% potable
- No version pinning in requirements.txt — consider pinning for reproducibility
