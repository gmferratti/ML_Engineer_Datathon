from typing import Dict
from src.train.constants import CLIENT_FEATURES, NEWS_FEATURES
import pandas as pd
import numpy as np
from src.recommendation_model.base_model import BaseRecommender
from sklearn.metrics import ndcg_score


def evaluate_model(
    model: BaseRecommender, evaluation_data: pd.DataFrame, k_ndcg: int = 10
) -> Dict:
    """
    Avalia o modelo de recomendação usando NDCG@k,
    considerando que evaluation_data possui colunas de features
    e uma coluna 'TARGET' com a relevância.

    Args:
        model (BaseRecommender): Modelo de recomendação treinado.
        evaluation_data (pd.DataFrame): DataFrame com colunas de features e a coluna 'TARGET'.
        k_ndcg (int, opcional): Corte para a métrica NDCG. Default: 10.

    Returns:
        Dict: Dicionário com as métricas de avaliação, ex.: {"NDCG@10": valor}.
    """
    # Separar X (features) e y (target)
    X = evaluation_data.drop(columns=["TARGET"])
    y_true = evaluation_data["TARGET"].values

    # Converter para numpy arrays
    client_feats = X[CLIENT_FEATURES].values
    news_feats = X[NEWS_FEATURES].values

    # Obter predições (scores) do modelo
    y_pred = model.predict({"client_features": client_feats, "news_features": news_feats})

    # Verifica que real e predito têm o mesmo tamanho
    assert len(y_true) == len(
        y_pred
    ), f"Dimensão incompatível: y_true tem {len(y_true)}, y_pred tem {len(y_pred)}"

    # Calcula o score usando objetos array-like
    ndcg_val = ndcg_score(np.array([y_true]), np.array([y_pred]), k=k_ndcg)

    metrics = {f"NDCG_{k_ndcg}": ndcg_val}
    return metrics
