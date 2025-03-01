import lightgbm as lgb
import numpy as np
from .base_model import BaseRecommender


class LightGBMRanker(BaseRecommender):
    def __init__(self, params=None, num_boost_round=100, **kwargs):
        """
        Inicializa o LightGBMRanker.

        Args:
            params (dict, optional): Parâmetros para LightGBM.
            num_boost_round (int, optional): Número de iterações.
            **kwargs: Parâmetros extras (por exemplo, threshold).
        """
        self.threshold = kwargs.pop("threshold", None)
        if params is not None and "num_class" in params:
            del params["num_class"]
        super().__init__(params=params, num_boost_round=num_boost_round)

    def train(self, X, y, group=None):
        """
        Treina o modelo em modo de ranking.

        Args:
            X: Features (array-like).
            y: Target (array-like).
            group: Grupo (array ou None).
        """
        if group is None:
            group = [X.shape[0]]
        else:
            if hasattr(group, "to_numpy"):
                group = group.to_numpy()
            if np.sum(group) != X.shape[0]:
                raise ValueError("Soma de 'group' diferente do total de linhas em X.")
        train_data = lgb.Dataset(X, label=y, group=group)
        self.model = lgb.train(self.params, train_data,
                               num_boost_round=self.num_boost_round)

    def predict(self, model_input):
        """
        Realiza a predição combinando as features.

        Args:
            model_input (dict): Contém 'client_features' e 'news_features'.

        Returns:
            np.ndarray: Scores preditos.
        """
        client_features = model_input.get("client_features")
        news_features = model_input.get("news_features")
        if client_features is None or news_features is None:
            raise ValueError("Input deve ter 'client_features' e 'news_features'.")
        X = np.concatenate([client_features, news_features], axis=1)
        if self.model is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        return self.model.predict(X)
