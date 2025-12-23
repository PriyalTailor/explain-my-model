class Explainer:
    """
    Lightweight, model-agnostic explainer for ML models.
    """

    def __init__(self, model, X_train):
        self.model = model
        self.X_train = X_train

    def global_feature_importance(self):
        raise NotImplementedError

    def explain_instance(self, instance):
        raise NotImplementedError

    def counterfactual(self, instance):
        raise NotImplementedError
