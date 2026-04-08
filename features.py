"""
Feature engineering for Water Potability Prediction.

Adds domain-specific features based on WHO water quality guidelines.
Used by both the training pipeline and the app at inference time.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

# WHO safe limits used for threshold-based binary flags
WHO_SAFE_RANGES = {
    "ph": (6.5, 8.5),
    "Hardness": (0, 300),
    "Chloramines": (0, 4),
    "Sulfate": (0, 250),
    "Conductivity": (0, 500),
    "Organic_carbon": (0, 2),
    "Trihalomethanes": (0, 80),
    "Turbidity": (0, 5),
}


class WaterFeatureEngineer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible transformer that adds engineered features.

    New features:
    - WHO threshold flags: binary indicators for each feature being within safe limits
    - who_violations_count: total number of WHO limit violations
    - Interaction features: ph_x_turbidity, chloramines_per_organic_carbon, solids_per_conductivity
    """

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        df = pd.DataFrame(X, columns=self._get_original_columns(X)) if not isinstance(X, pd.DataFrame) else X.copy()

        # WHO threshold flags
        for feat, (lo, hi) in WHO_SAFE_RANGES.items():
            if feat in df.columns:
                df[f"{feat}_safe"] = ((df[feat] >= lo) & (df[feat] <= hi)).astype(int)

        # Count of WHO violations
        safe_cols = [c for c in df.columns if c.endswith("_safe")]
        if safe_cols:
            df["who_violations_count"] = len(safe_cols) - df[safe_cols].sum(axis=1)

        # Interaction features
        if "ph" in df.columns and "Turbidity" in df.columns:
            df["ph_x_turbidity"] = df["ph"] * df["Turbidity"]

        if "Chloramines" in df.columns and "Organic_carbon" in df.columns:
            df["chloramines_per_organic_carbon"] = df["Chloramines"] / (df["Organic_carbon"] + 1e-8)

        if "Solids" in df.columns and "Conductivity" in df.columns:
            df["solids_per_conductivity"] = df["Solids"] / (df["Conductivity"] + 1e-8)

        return df

    def _get_original_columns(self, X):
        """Get column names, handling both DataFrames and arrays."""
        if hasattr(X, "columns"):
            return list(X.columns)
        return [
            "ph", "Hardness", "Solids", "Chloramines", "Sulfate",
            "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity",
        ]

    def get_feature_names_out(self, input_features=None):
        """Return output feature names (required for sklearn Pipeline compatibility)."""
        dummy = pd.DataFrame(
            np.zeros((1, 9)),
            columns=["ph", "Hardness", "Solids", "Chloramines", "Sulfate",
                      "Conductivity", "Organic_carbon", "Trihalomethanes", "Turbidity"]
        )
        result = self.transform(dummy)
        return list(result.columns)
