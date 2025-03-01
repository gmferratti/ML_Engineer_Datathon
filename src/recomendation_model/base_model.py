# src/recomendation_model/base_model.py
from abc import ABC, abstractmethod
import lightgbm as lgb
import pandas as pd
import numpy as np
import os

class BaseRecommender(ABC):
    def __init__(self, params=None, num_boost_round=100):
        """
        Inicializa o modelo com parâmetros padrão ou personalizados para ranking.

        Args:
            params (dict, opcional): Parâmetros para o LightGBM. Caso não seja fornecido,
                                     utiliza um conjunto padrão voltado para ranking.
            num_boost_round (int, opcional): Número de iterações (árvores) no treinamento.
        """
        # Define os parâmetros padrão para ranking, incluindo label_gain para scores de 0 a 100
        default_params = {
            "objective": "lambdarank",  # Usando objetivo de ranking (LambdaRank)
            "metric": "ndcg",  # Métrica típica para avaliar ranking
            "learning_rate": 0.05,
            "num_leaves": 39,
            "verbose": -1,
            "label_gain": [
                2**i - 1 for i in range(101)
            ],  # Vetor de ganho para rótulos de 0 a 100
        }

        # Se o usuário passar parâmetros personalizados, verifica se label_gain não está definido
        if params is not None:
            if (
                params.get("objective", default_params["objective"]) == "lambdarank"
                and "label_gain" not in params
            ):
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
        Este método é abstrato aqui, mas vamos ilustrar abaixo como pode ser implementado.

        Args:
            model_input (dict): Dicionário que deve conter as chaves
                                'client_features' e 'news_features'.
        """
        pass

    @abstractmethod
    def train(self, X, y):
        """
        Realiza o treinamento do modelo.
        Método abstrato para ser implementado.
        """
        pass


class LightGBMRanker(BaseRecommender):
    """
    Exemplo concreto de implementação de um modelo de ranking no LightGBM,
    utilizando o objetivo 'lambdarank' para aprendizado de ranking.
    
    Esta classe aceita parâmetros extras via **kwargs, permitindo a utilização
    de um 'threshold' para ajustar os scores preditos.
    """
    def __init__(self, params=None, num_boost_round=100, **kwargs):
        self.threshold = kwargs.pop('threshold', None)
        if params is not None and "num_class" in params:
            del params["num_class"]
        super().__init__(params=params, num_boost_round=num_boost_round)

    def train(self, X, y, group=pd.read_parquet("C:/Users/gufer/OneDrive/Documentos/FIAP/Fase_05/ML_Engineer_Datathon/data/train/group_train.parquet")):
        """
        Treina o modelo LightGBM em modo de ranking (LambdaRank).

        Args:
            X (array-like): Features de todos os pares (usuário-notícia),
                            por exemplo [n_amostras x n_features].
            y (array-like): Alvo (score ou relevância) de cada par (usuário-notícia).
            group (list, array ou DataFrame/Series, opcional): Quantidade de amostras 
                            (instâncias) para cada usuário ou grupo. Se não for informado, 
                            assume que todas as instâncias pertencem a um único grupo.
        """
        # Se group não for informado, assume que todos os samples pertencem a um único grupo
        if group is None:
            group = [X.shape[0]]
        else:
            # Se o group for um DataFrame ou Series, converte para um array NumPy
            if isinstance(group, pd.DataFrame):
                # Se houver uma coluna chamada "groupCount", a utiliza; caso contrário, pega a primeira coluna
                if "groupCount" in group.columns:
                    group = group["groupCount"].to_numpy()
                else:
                    group = group.iloc[:, 0].to_numpy()
            elif isinstance(group, pd.Series):
                group = group.to_numpy()

            # Agora que group é um array NumPy, a soma é realizada sem ambiguidade
            if np.sum(group) != X.shape[0]:
                raise ValueError("A soma dos valores em 'group' deve ser igual ao número de amostras em X.")

        # Cria Dataset para ranking, informando os grupos
        train_data = lgb.Dataset(X, label=y, group=group)

        # Treina o modelo com os parâmetros definidos
        self.model = lgb.train(self.params, train_data, num_boost_round=self.num_boost_round)

    def predict(self, model_input):
        """
        Realiza a predição (score) para ranqueamento.
        
        Suporta os seguintes formatos de input:
        - Se model_input for um DataFrame, assume-se que ele contém todas as features
            na ordem definida na assinatura (por exemplo, 'isWeekend', 'relLocalState', etc.)
        - Se model_input for um dicionário, verifica se contém as chaves
            'client_features' e 'news_features' (formato antigo) e as concatena.
        
        Args:
            model_input (dict ou pd.DataFrame): Dados de entrada.
        
        Returns:
            np.ndarray: Array com os scores preditos (quanto maior o score,
                        maior a relevância/rank).
        """
        client_features = model_input.get("client_features")
        news_features = model_input.get("news_features")

        if client_features is None or news_features is None:
            raise ValueError(
                "O dicionário 'model_input' deve conter as chaves "
                "'client_features' e 'news_features'."
            )

        # Combina as features do usuário e da notícia
        X = np.concatenate([client_features, news_features], axis=1)

        if self.model is None:
            raise ValueError(
                "O modelo ainda não foi treinado. Execute train() antes de predict()."
            )

        scores = self.model.predict(X)
        return scores
