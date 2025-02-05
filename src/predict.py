from data.data_loader import (
    get_client_features,
    get_non_viewed_news,
    get_predicted_news
)
import pandas as pd
from typing import List
from recomendation_model.base_model import BaseRecommender


def predict_for_user_id(
    user_id: str,
    news_features_df: pd.DataFrame,
    clients_features_df: pd.DataFrame,
    model: BaseRecommender,
    n: int = 5,
    score_threshold: float = 0.3
) -> List[str]:
    """
    Realiza a predição para um usuario especifico.

    Args:
        user_id (str): ID do usuario a ser previsto.
        news_features_df (DataFrame): Noticias validas para recomendacao.
        clients_features_df (DataFrame): Features dos clientes.
        model: Instancia do modelo de recomendacao com metodo predict.
        n (int): Quantidade de noticias a recomendar (default: 5).
        score_threshold (float): Score minimo para considerar a recomendacao.

    Returns:
        List[str]: Lista de IDs das noticias recomendadas.
    """
    client_features = get_client_features(user_id, clients_features_df)
    non_viewed_news_features = get_non_viewed_news(user_id, news_features_df)

    scores = model.predict(client_features, non_viewed_news_features)

    predicted_news = get_predicted_news(
        scores,
        non_viewed_news_features,
        n=n,
        score_threshold=score_threshold
    )

    return predicted_news
