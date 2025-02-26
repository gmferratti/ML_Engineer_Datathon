from typing import List

import pandas as pd

from data.data_loader import get_client_features, get_non_viewed_news, get_predicted_news


def predict_for_userId(
    userId: str,
    news_features_df: pd.DataFrame,
    clients_features_df: pd.DataFrame,
    model,
    n: int = 5,
    score_threshold: float = 0.3,
) -> List[str]:
    # Obtém features combinadas de forma dinâmica
    client_features = get_client_features(userId, clients_features_df).to_frame().T
    non_viewed_news = get_non_viewed_news(userId, news_features_df, clients_features_df)

    # Para cada noticia combina com as features do usuario
    model_input = non_viewed_news.assign(userId=userId).merge(
        client_features.drop(columns=["userId"]), how="cross", suffixes=("_news", "_user")
    )

    scores = model.predict(model_input)
    return get_predicted_news(scores, non_viewed_news, n=n, score_threshold=score_threshold)
