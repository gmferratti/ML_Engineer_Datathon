# src/recomendation_model/base_model.py
from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    @abstractmethod
    def predict(self, model_input):
        """
        Realiza a predicao.

        Args:
            model_input: Dicionário com client_features e news_features
        """
        pass

    @abstractmethod
    def train(self, X, y):
        """Realiza o treinamento do modelo."""
        pass
