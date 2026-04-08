# Water-Potability-Prediction

This project predicts **water potability** using **machine learning** based on key chemical properties such as **pH, hardness, conductivity, and organic carbon**. It applies **data preprocessing, outlier handling, and model training** to assess whether water is **safe for consumption**.

## What We Are Trying to Achieve
The goal of this project is to develop an **efficient and accurate model** to determine water quality. By analyzing various water parameters, we classify whether the water is **potable (safe) or not**. This project provides an **ML-based solution** to help in water safety assessment.

## Technologies Used
- **Pandas & NumPy**: Data cleaning, preprocessing, and handling missing values.
- **Matplotlib & Seaborn**: Exploratory Data Analysis (EDA) and visualization.
- **Scikit-Learn**: Machine learning model training and evaluation.
- **Random Forest Classifier**: Primary algorithm for water quality prediction.
- **Pickle**: Model serialization for future use.

## Dataset
The dataset contains key water quality indicators:
- **pH**: Acidity/alkalinity level of water.
- **Hardness**: Concentration of minerals in water.
- **Conductivity**: Water’s ability to conduct electricity.
- **Organic Carbon & Trihalomethanes**: Indicators of organic contaminants.
- **Potability**: Target variable (1 = potable, 0 = non-potable).

## Model Performance
- **Train-test split:** 80-20 ratio.
- **Best accuracy achieved:** ~80% using **Random Forest**.
- **Handling Outliers:** IQR method applied to **Solids** column.

## Installation
1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/Water-Potability-Prediction.git
