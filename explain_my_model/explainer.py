import shap
import pandas as pd
import numpy as np

class Explainer:
    """
    Lightweight, model-agnostic explainer for ML models.
    """

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

        # SHAP explainer (Tree-based for RandomForest)
        self.explainer = shap.TreeExplainer(self.model)

    def global_feature_importance(self, max_display=10):
        """
        Returns global feature importance using mean absolute SHAP values.
        Robust to all modern SHAP output formats.
        """

        shap_output = self.explainer(self.X_train)

        # Extract raw SHAP values safely
        if hasattr(shap_output, "values"):
            shap_values = shap_output.values
        else:
            shap_values = shap_output

        # Handle multiclass / binary shapes
        if shap_values.ndim == 3:
            # (samples, features, classes) → take positive class
            shap_values = shap_values[:, :, 1]

        # Ensure 2D: (samples, features)
        shap_values = np.array(shap_values)

        importance = np.abs(shap_values).mean(axis=0)

        importance_df = pd.DataFrame({
            "feature": list(self.X_train.columns),
            "importance": importance.flatten()
        })

        return (
            importance_df
            .sort_values(by="importance", ascending=False)
            .head(max_display)
        )


