import os
import pickle
import streamlit as st
import pandas as pd

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

# Function to Load the Model
@st.cache_resource
def load_model():
    model_path = os.path.join(os.path.dirname(__file__), 'models', 'water_potability_random_forest.pkl')
    if os.path.exists(model_path):
        return pickle.load(open(model_path, 'rb'))
    else:
        st.error("Model file not found in 'models/' directory. Please check the path.")
        return None

model = load_model()

# Main App Logic
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

    # Predict Button and Logic
    if st.button('Predict Potability', type='primary'):
        try:
            # Prepare input data as a DataFrame with correct feature names
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

            # Check for all-zero inputs
            if all(value == 0 for value in input_data.iloc[0]):
                st.markdown("""
                    <div style='background-color: #FFE4E1; padding: 20px; border-radius: 10px;'>
                        <h3 style='color: #FF6B6B; margin: 0;'>⚠️ Water is Not Potable</h3>
                        <p style='color: #FF3333;'>The water is unsafe for drinking.</p>
                    </div>
                """, unsafe_allow_html=True)
            else:
                # Make prediction using the loaded model
                prediction = model.predict(input_data)[0]
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

    # Expandable Information Panel (Moved to the End)
    with st.expander("About This App"):
        st.markdown("""
        **Developed by:** Dhruv Jaradi and Kevin Dave  
        **Dataset Used:** Water Potability Dataset (source: [Kaggle](https://www.kaggle.com/datasets/adityakadiwal/water-potability))  
        **Machine Learning Algorithm:** Random Forest Classifier  
        **Description:** This app uses a Random Forest model trained on physicochemical water features to predict potability. 
        """)
