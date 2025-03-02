import pandas as pd
from mlflow.pyfunc import PythonModel

from src.recommendation_model.base_model import BaseRecommender


class MockedRecommender(BaseRecommender):
    def __init__(self, **kwargs):
        self.params = kwargs

    def predict(self, model_input: pd.DataFrame):
        """Gera scores baseado nas colunas combinadas"""
        return [0.5 for _ in range(len(model_input))]

    def train(self, X, y):
        pass


class MLflowWrapper(PythonModel):
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input: pd.DataFrame):
        return self.model.predict(model_input)
