import json
import os
import pickle

import pandas as pd
import streamlit as st

# Custom Header with Styling
st.markdown("""
    <div class="header-container">
        <h1>Water Potability Prediction</h1>
        <p>This application predicts the potability of water based on key physicochemical features.
        Simply enter the required values below and let the app determine whether the water is potable.</p>
    </div>
    <style>
        .header-container {
            background-color: #f0f2f6;
            padding: 20px;
            border-radius: 10px;
            text-align: center;
        }
        .header-container h1 {
            color: #4CAF50;
            margin-bottom: 10px;
        }
        .header-container p {
            color: #333;
            font-size: 16px;
        }
    </style>
""", unsafe_allow_html=True)

# Add Space and Heading
st.markdown("""
    <div style="height: 40px;"></div>
    <h2 style="color: #4CAF50; text-align: center;">Enter Water Quality Features</h2>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model():
    """Load the best trained model pipeline."""
    base = os.path.dirname(__file__)
    # Prefer the new best_model.pkl; fall back to legacy model
    model_path = os.path.join(base, "models", "best_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(base, "models", "water_potability_random_forest.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    st.error("Model file not found in 'models/' directory. Please check the path.")
    return None


@st.cache_data
def load_model_info():
    """Load model comparison metrics if available."""
    path = os.path.join(os.path.dirname(__file__), "models", "model_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


model = load_model()
model_info = load_model_info()

# Show which model is loaded
if model_info:
    best = model_info["best_model"]
    metrics = model_info["models"][best]
    st.caption(
        f"Model: **{best}** | F1: {metrics['test_f1']:.3f} | "
        f"ROC-AUC: {metrics['test_roc_auc']:.3f} | "
        f"Accuracy: {metrics['test_accuracy']:.3f}"
    )

if model:
    # Create a 3-column layout for input fields
    col1, col2, col3 = st.columns(3)

    with col1:
        ph = st.number_input('pH Level', min_value=0.0, max_value=14.0, value=7.0, help="Typical range: 6.5 - 8.5")
        hardness = st.number_input('Hardness (mg/L)', value=150.0)
        solids = st.number_input('Solids (mg/L)', value=500.0)

    with col2:
        chloramines = st.number_input('Chloramines (ppm)', value=2.0)
        sulfate = st.number_input('Sulfate (mg/L)', value=100.0)
        conductivity = st.number_input('Conductivity (μS/cm)', value=400.0)

    with col3:
        organic_carbon = st.number_input('Organic Carbon (mg/L)', value=1.0)
        trihalomethanes = st.number_input('Trihalomethanes (μg/L)', value=50.0)
        turbidity = st.number_input('Turbidity (NTU)', value=1.0)

    # WHO safe limits for input warnings
    WHO_LIMITS = {
        "ph": (6.5, 8.5),
        "Hardness": (None, 300),
        "Chloramines": (None, 4),
        "Sulfate": (None, 250),
        "Conductivity": (None, 500),
        "Organic_carbon": (None, 2),
        "Trihalomethanes": (None, 80),
        "Turbidity": (None, 5),
    }

    # Predict Button and Logic
    if st.button('Predict Potability', type='primary'):
        try:
            input_data = pd.DataFrame({
                'ph': [ph],
                'Hardness': [hardness],
                'Solids': [solids],
                'Chloramines': [chloramines],
                'Sulfate': [sulfate],
                'Conductivity': [conductivity],
                'Organic_carbon': [organic_carbon],
                'Trihalomethanes': [trihalomethanes],
                'Turbidity': [turbidity]
            })

            # Input validation: warn about WHO limit violations
            warnings_list = []
            values = {"ph": ph, "Hardness": hardness, "Chloramines": chloramines,
                      "Sulfate": sulfate, "Conductivity": conductivity,
                      "Organic_carbon": organic_carbon, "Trihalomethanes": trihalomethanes,
                      "Turbidity": turbidity}
            for feat, (lo, hi) in WHO_LIMITS.items():
                val = values[feat]
                if lo is not None and val < lo:
                    warnings_list.append(f"{feat} ({val}) is below WHO minimum ({lo})")
                if hi is not None and val > hi:
                    warnings_list.append(f"{feat} ({val}) exceeds WHO limit ({hi})")

            if warnings_list:
                st.warning("**WHO limit warnings:**\n" + "\n".join(f"- {w}" for w in warnings_list))

            if all(value == 0 for value in input_data.iloc[0]):
                st.markdown("""
                    <div style='background-color: #FFE4E1; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #FF6B6B; margin: 0;'>⚠️ Water is Not Potable</h3>
                        <p style='color: #FF3333;'>All values are zero — the water is unsafe for drinking.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                prediction = model.predict(input_data)[0]

                # Show probability if model supports it
                confidence = None
                if hasattr(model, "predict_proba"):
                    proba = model.predict_proba(input_data)[0]
                    confidence = proba[1]  # probability of potable

                if prediction == 1:
                    st.markdown("""
                        <div style='background-color: #E1FFE4; padding: 20px; border-radius: 10px;'>
                            <h3 style='color: #6BFF6B; margin: 0;'>✅ Water is Potable</h3>
                            <p style='color: #4CAF50;'>The water is safe to drink.</p>
                        </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown("""
                        <div style='background-color: #FFE4E1; padding: 20px; border-radius: 10px;'>
                            <h3 style='color: #FF6B6B; margin: 0;'>⚠️ Water is Not Potable</h3>
                            <p style='color: #FF3333;'>The water is unsafe for drinking.</p>
                        </div>
                    """, unsafe_allow_html=True)

                if confidence is not None:
                    st.markdown(f"<div style='height: 10px;'></div>", unsafe_allow_html=True)
                    st.metric("Potability Confidence", f"{confidence:.1%}")
                    st.progress(confidence)

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")

    # Additional Information Expander (Features and Limits)
    with st.expander("About this predictor"):
        st.markdown("""
        This water potability prediction model uses a machine learning algorithm trained on water quality data.
        The model evaluates water safety based on key chemical and physical properties.

        **Features used and recommended limits:**
        - **pH Level:** 6.5 - 8.5 (Safe range)
        - **Hardness:** ≤ 300 mg/L
        - **Solids:** No strict limit, total dissolved solids
        - **Chloramines:** ≤ 4 ppm
        - **Sulfate:** ≤ 250 mg/L
        - **Conductivity:** ≤ 500 μS/cm
        - **Organic Carbon:** ≤ 2 mg/L
        - **Trihalomethanes:** ≤ 80 μg/L
        - **Turbidity:** ≤ 5 NTU
        """)

    # Model comparison table
    if model_info:
        with st.expander("Model Comparison Results"):
            rows = []
            for name, m in model_info["models"].items():
                rows.append({
                    "Model": name,
                    "F1": f"{m['test_f1']:.4f}",
                    "ROC-AUC": f"{m['test_roc_auc']:.4f}",
                    "Precision": f"{m['test_precision']:.4f}",
                    "Recall": f"{m['test_recall']:.4f}",
                    "Accuracy": f"{m['test_accuracy']:.4f}",
                    "CV F1": f"{m['cv_best_f1']:.4f}",
                })
            st.table(pd.DataFrame(rows))
            st.caption(f"Best model selected: **{model_info['best_model']}** (by F1 score)")

    with st.expander("About This App"):
        st.markdown("""
        **Developed by:** Dhruv Jaradi and Kevin Dave
        **Dataset Used:** Water Potability Dataset (source: [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability))
        **Machine Learning Algorithm:** Multi-model comparison (RF, XGBoost, LightGBM, Gradient Boosting) with SMOTE + hyperparameter tuning
        **Description:** This app uses the best-performing model from a comparative evaluation, trained on physicochemical water features to predict potability.
        """)
