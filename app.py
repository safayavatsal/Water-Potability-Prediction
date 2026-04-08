import json
import os
import pickle

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

st.set_page_config(page_title="Water Potability Prediction", page_icon="💧", layout="wide")

# ── Shared helpers ──────────────────────────────────────────────────────────

WHO_LIMITS = {
    "ph": {"min": 6.5, "max": 8.5, "unit": ""},
    "Hardness": {"min": 0, "max": 300, "unit": "mg/L"},
    "Solids": {"min": 0, "max": 500, "unit": "mg/L"},
    "Chloramines": {"min": 0, "max": 4, "unit": "ppm"},
    "Sulfate": {"min": 0, "max": 250, "unit": "mg/L"},
    "Conductivity": {"min": 0, "max": 500, "unit": "μS/cm"},
    "Organic_carbon": {"min": 0, "max": 2, "unit": "mg/L"},
    "Trihalomethanes": {"min": 0, "max": 80, "unit": "μg/L"},
    "Turbidity": {"min": 0, "max": 5, "unit": "NTU"},
}


@st.cache_resource
def load_model():
    base = os.path.dirname(__file__)
    model_path = os.path.join(base, "models", "best_model.pkl")
    if not os.path.exists(model_path):
        model_path = os.path.join(base, "models", "water_potability_random_forest.pkl")
    if os.path.exists(model_path):
        with open(model_path, "rb") as f:
            return pickle.load(f)
    return None


@st.cache_data
def load_model_info():
    path = os.path.join(os.path.dirname(__file__), "models", "model_comparison.json")
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


def radar_chart(input_values: dict) -> go.Figure:
    """Create a radar chart comparing user input to WHO safe limits."""
    features = list(WHO_LIMITS.keys())
    # Normalize: value / WHO max limit (so 1.0 = at the limit)
    normalized_input = []
    normalized_limit = []
    labels = []
    for feat in features:
        limit = WHO_LIMITS[feat]["max"]
        val = input_values.get(feat, 0)
        normalized_input.append(val / limit if limit else 0)
        normalized_limit.append(1.0)
        labels.append(feat)

    # Close the polygon
    normalized_input.append(normalized_input[0])
    normalized_limit.append(normalized_limit[0])
    labels.append(labels[0])

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=normalized_limit, theta=labels, fill="toself",
        name="WHO Safe Limit", opacity=0.2,
        line=dict(color="#4CAF50", dash="dash"),
    ))
    fig.add_trace(go.Scatterpolar(
        r=normalized_input, theta=labels, fill="toself",
        name="Your Sample", opacity=0.5,
        line=dict(color="#2196F3"),
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(max(normalized_input), 1.5)])),
        showlegend=True, height=450,
        title="Your Water Sample vs WHO Safe Limits",
    )
    return fig


# ── Page definitions ────────────────────────────────────────────────────────

def page_predict():
    st.markdown("""
        <div style="background-color: #f0f2f6; padding: 20px; border-radius: 10px; text-align: center;">
            <h1 style="color: #4CAF50; margin-bottom: 10px;">Water Potability Prediction</h1>
            <p style="color: #333; font-size: 16px;">Enter water quality parameters to predict whether the water is safe to drink.</p>
        </div>
    """, unsafe_allow_html=True)

    model = load_model()
    model_info = load_model_info()

    if model_info:
        best = model_info["best_model"]
        m = model_info["models"][best]
        st.caption(f"Model: **{best}** | F1: {m['test_f1']:.3f} | ROC-AUC: {m['test_roc_auc']:.3f} | Accuracy: {m['test_accuracy']:.3f}")

    if not model:
        st.error("Model file not found. Please run `python train_model.py` first.")
        return

    st.markdown("<div style='height: 20px;'></div>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        ph = st.number_input("pH Level", min_value=0.0, max_value=14.0, value=7.0, help="Typical range: 6.5 - 8.5")
        hardness = st.number_input("Hardness (mg/L)", value=150.0)
        solids = st.number_input("Solids (mg/L)", value=500.0)
    with col2:
        chloramines = st.number_input("Chloramines (ppm)", value=2.0)
        sulfate = st.number_input("Sulfate (mg/L)", value=100.0)
        conductivity = st.number_input("Conductivity (μS/cm)", value=400.0)
    with col3:
        organic_carbon = st.number_input("Organic Carbon (mg/L)", value=1.0)
        trihalomethanes = st.number_input("Trihalomethanes (μg/L)", value=50.0)
        turbidity = st.number_input("Turbidity (NTU)", value=1.0)

    input_values = {
        "ph": ph, "Hardness": hardness, "Solids": solids,
        "Chloramines": chloramines, "Sulfate": sulfate, "Conductivity": conductivity,
        "Organic_carbon": organic_carbon, "Trihalomethanes": trihalomethanes, "Turbidity": turbidity,
    }

    if st.button("Predict Potability", type="primary"):
        try:
            input_data = pd.DataFrame({k: [v] for k, v in input_values.items()})

            # WHO warnings
            warnings_list = []
            for feat, limits in WHO_LIMITS.items():
                val = input_values[feat]
                if "min" in limits and limits["min"] and val < limits["min"]:
                    warnings_list.append(f"{feat} ({val}) is below WHO minimum ({limits['min']})")
                if "max" in limits and val > limits["max"]:
                    warnings_list.append(f"{feat} ({val}) exceeds WHO limit ({limits['max']})")
            if warnings_list:
                st.warning("**WHO limit warnings:**\n" + "\n".join(f"- {w}" for w in warnings_list))

            if all(v == 0 for v in input_values.values()):
                st.error("All values are zero — cannot make a meaningful prediction.")
            else:
                prediction = model.predict(input_data)[0]
                confidence = None
                if hasattr(model, "predict_proba"):
                    confidence = model.predict_proba(input_data)[0][1]

                if prediction == 1:
                    st.success("**✅ Water is Potable** — The water is safe to drink.")
                else:
                    st.error("**⚠️ Water is Not Potable** — The water is unsafe for drinking.")

                if confidence is not None:
                    st.metric("Potability Confidence", f"{confidence:.1%}")
                    st.progress(confidence)

                # Radar chart
                st.plotly_chart(radar_chart(input_values), use_container_width=True)

                # SHAP waterfall
                with st.spinner("Generating explanation..."):
                    try:
                        from explainability import compute_shap_values, load_data, plot_waterfall
                        X_bg = load_data()
                        shap_vals = compute_shap_values(model, X_bg, input_data)
                        fig = plot_waterfall(shap_vals, index=0)
                        st.subheader("Why did the model make this prediction?")
                        st.pyplot(fig)
                    except Exception:
                        pass

        except Exception as e:
            st.error(f"An error occurred during prediction: {str(e)}")


def page_explore():
    st.header("Data Exploration")
    data_path = os.path.join(os.path.dirname(__file__), "data", "water_potability.csv")
    df = pd.read_csv(data_path)

    st.subheader("Dataset Overview")
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Samples", len(df))
    col2.metric("Potable", int(df["Potability"].sum()))
    col3.metric("Non-Potable", int((df["Potability"] == 0).sum()))

    st.subheader("Feature Distributions")
    feature = st.selectbox("Select feature", df.columns.drop("Potability"))
    import plotly.express as px
    fig = px.histogram(df, x=feature, color="Potability", barmode="overlay",
                       color_discrete_map={0: "#FF6B6B", 1: "#4CAF50"},
                       labels={"Potability": "Potable"})
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation Heatmap")
    corr = df.corr(numeric_only=True)
    fig_corr = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale="RdBu_r", zmin=-1, zmax=1, text=corr.values.round(2),
        texttemplate="%{text}", textfont={"size": 10},
    ))
    fig_corr.update_layout(height=500, title="Feature Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Statistical Summary")
    st.dataframe(df.describe().round(2))

    st.subheader("Missing Values")
    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing) > 0:
        st.dataframe(pd.DataFrame({"Feature": missing.index, "Missing Count": missing.values,
                                    "% Missing": (missing.values / len(df) * 100).round(1)}))
    else:
        st.info("No missing values in dataset.")


def page_model_performance():
    st.header("Model Performance")
    model_info = load_model_info()

    if not model_info:
        st.warning("No model comparison data found. Run `python train_model.py` to generate it.")
        return

    best_name = model_info["best_model"]

    # Comparison table
    st.subheader("Model Comparison")
    rows = []
    for name, m in model_info["models"].items():
        rows.append({
            "Model": ("🏆 " + name) if name == best_name else name,
            "F1": round(m["test_f1"], 4),
            "ROC-AUC": round(m["test_roc_auc"], 4),
            "Precision": round(m["test_precision"], 4),
            "Recall": round(m["test_recall"], 4),
            "Accuracy": round(m["test_accuracy"], 4),
            "CV F1": round(m["cv_best_f1"], 4),
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Bar chart comparison
    metrics_to_plot = ["test_f1", "test_roc_auc", "test_precision", "test_recall", "test_accuracy"]
    chart_data = []
    for name, m in model_info["models"].items():
        for metric in metrics_to_plot:
            chart_data.append({"Model": name, "Metric": metric.replace("test_", ""), "Value": m[metric]})
    import plotly.express as px
    fig = px.bar(pd.DataFrame(chart_data), x="Metric", y="Value", color="Model", barmode="group",
                 color_discrete_sequence=["#4CAF50", "#2196F3", "#FF9800", "#9C27B0"])
    fig.update_layout(height=400, title="Model Metrics Comparison", yaxis_range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)

    # SHAP global importance
    st.subheader("Global Feature Importance (SHAP)")
    with st.spinner("Computing SHAP values..."):
        try:
            from explainability import load_data, plot_beeswarm, plot_global_feature_importance
            model = load_model()
            X_bg = load_data()
            col1, col2 = st.columns(2)
            with col1:
                fig_bar = plot_global_feature_importance(model, X_bg)
                st.pyplot(fig_bar)
            with col2:
                fig_bee = plot_beeswarm(model, X_bg)
                st.pyplot(fig_bee)
        except Exception as e:
            st.info(f"SHAP plots unavailable: {e}")


def page_about():
    st.header("About")
    st.markdown("""
    ### Water Potability Prediction

    This application predicts whether water is safe for human consumption based on 9 physicochemical properties.

    **Features used and WHO recommended limits:**
    | Feature | Safe Limit | Unit |
    |---------|-----------|------|
    | pH Level | 6.5 - 8.5 | - |
    | Hardness | ≤ 300 | mg/L |
    | Solids | No strict limit | mg/L |
    | Chloramines | ≤ 4 | ppm |
    | Sulfate | ≤ 250 | mg/L |
    | Conductivity | ≤ 500 | μS/cm |
    | Organic Carbon | ≤ 2 | mg/L |
    | Trihalomethanes | ≤ 80 | μg/L |
    | Turbidity | ≤ 5 | NTU |

    **Developed by:** Dhruv Jaradi and Kevin Dave

    **Dataset:** [Water Potability Dataset (Kaggle)](https://www.kaggle.com/datasets/adityakadiwal/water-potability)

    **ML Pipeline:** Multi-model comparison (Random Forest, XGBoost, LightGBM, Gradient Boosting)
    with SMOTE oversampling, StandardScaler, and hyperparameter tuning via GridSearchCV.
    """)


# ── Navigation ──────────────────────────────────────────────────────────────

pages = {
    "Predict": page_predict,
    "Data Exploration": page_explore,
    "Model Performance": page_model_performance,
    "About": page_about,
}

st.sidebar.title("💧 Navigation")
selection = st.sidebar.radio("Go to", list(pages.keys()))
pages[selection]()
