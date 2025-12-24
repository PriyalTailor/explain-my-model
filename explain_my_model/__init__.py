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

        # Use unified SHAP API (handles shape issues)
        self.explainer = shap.Explainer(self.model, self.X_train)
