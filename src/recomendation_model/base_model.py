from abc import ABC, abstractmethod


class BaseRecommender(ABC):
    def __init__(self, params=None, num_boost_round=100):
        """
        Inicializa o modelo com parâmetros para ranking.

        Args:
            params (dict, optional): Parâmetros para LightGBM.
            num_boost_round (int, optional): Número de iterações.
        """
        default_params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "learning_rate": 0.05,
            "num_leaves": 39,
            "verbose": -1,
            "label_gain": [2**i - 1 for i in range(101)]
        }
        if params is not None:
            if (params.get("objective", default_params["objective"]) ==
                    "lambdarank" and "label_gain" not in params):
                params["label_gain"] = [2**i - 1 for i in range(101)]
            self.params = params
        else:
            self.params = default_params
        self.num_boost_round = num_boost_round
        self.model = None

    @abstractmethod
    def predict(self, model_input):
        """
        Realiza a predição.

        Args:
            model_input (dict): Contém 'client_features' e 'news_features'.

        Returns:
            np.ndarray: Scores preditos.
        """
        pass

    @abstractmethod
    def train(self, X, y):
        """
        Treina o modelo.

        Args:
            X: Features.
            y: Target.
        """
        pass
