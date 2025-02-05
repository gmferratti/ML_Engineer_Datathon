from recomendation_model.base_model import BaseRecommender


class MockedRecommender(BaseRecommender):
    def __init__(self, **kwargs):
        # Inicialize parâmetros ou carregue modelo treinado
        self.params = kwargs

    def predict(self, client_features, news_features):
        # Exemplo simples (pode ser mockado para experimentação)
        scores = [0.5 for _ in news_features]
        return scores

    def train(self, X, y):
        # Lógica de treinamento (pode ser um mock durante prototipação)
        pass
