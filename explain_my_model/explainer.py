import shap
import pandas as pd
import numpy as np

class Explainer:
    """
    Lightweight, model-agnostic explainer for ML models.
    """

    def __init__(self, model, X_train):
        if model is None:
            raise ValueError("Model must not be None")
        if X_train is None or len(X_train) == 0:
            raise ValueError("X_train must contain data")

        self.model = model
        self.X_train = X_train
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
    
    def explain_instance(self, instance):
        """
        Explains a single prediction using SHAP values.
        Robust to all SHAP output shapes.
        """
        instance_df = pd.DataFrame([instance], columns=self.X_train.columns)

        shap_values = self.explainer(instance_df)

        values = shap_values.values

        # ---- SHAPE HANDLING (CRITICAL) ----
        # Possible shapes:
        # (1, n_features)
        # (1, n_features, n_classes)
        # (1, n_classes, n_features)

        if values.ndim == 3:
            # If shape is (1, n_features, n_classes)
            if values.shape[1] == len(self.X_train.columns):
                values = values[0, :, 1]   # take positive class
            else:
                values = values[0, 1, :]   # alternative layout
        else:
            values = values[0]

        values = values.flatten()  # ensure 1D

        explanation = pd.DataFrame({
            "feature": self.X_train.columns,
            "shap_value": values
        }).sort_values(by="shap_value", ascending=False)

        return explanation
    
    def explain_instance_text(self, instance, top_k=5):
        """
        Returns a human-readable explanation.
        """
        explanation = self.explain_instance(instance).head(top_k)

        text = "Top factors influencing the prediction:\n"
        for _, row in explanation.iterrows():
            direction = "increased" if row["shap_value"] > 0 else "decreased"
            text += f"- {row['feature']} {direction} the risk\n"

        return text
    
    def counterfactual(self, instance, step=0.1, max_iter=50):
        """
        Generates a simple counterfactual by perturbing features.
        """
        instance_cf = instance.copy()
        original_pred = self.model.predict(pd.DataFrame([instance], columns=self.X_train.columns))[0]

        for _ in range(max_iter):
            for feature in self.X_train.columns:
                # Try decreasing feature
                instance_cf[feature] -= step
                new_pred = self.model.predict(
                    pd.DataFrame([instance_cf], columns=self.X_train.columns)
                )[0]

                if new_pred != original_pred:
                    return instance_cf

                # Try increasing feature
                instance_cf[feature] += 2 * step
                new_pred = self.model.predict(
                    pd.DataFrame([instance_cf], columns=self.X_train.columns)
                )[0]

                if new_pred != original_pred:
                    return instance_cf

                # Restore original
                instance_cf[feature] -= step

        return None
    
    def predict(self, instance):
        """
        Returns model prediction for a single instance.
        """
        instance_df = pd.DataFrame([instance], columns=self.X_train.columns)
        return self.model.predict(instance_df)[0]









