from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    @abstractmethod
    def predict(self, client_features, news_features):
        """Realiza a predicao."""
        pass

    @abstractmethod
    def train(self, X, y):
        """Realiza o treinamento do modelo."""
        pass
