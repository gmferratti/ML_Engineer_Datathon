from typing import List

import pandas as pd

from data.data_loader import get_client_features, get_non_viewed_news, get_predicted_news


def predict_for_user_id(
    user_id: str,
    news_features_df: pd.DataFrame,
    clients_features_df: pd.DataFrame,
    model,
    n: int = 5,
    score_threshold: float = 0.3,
) -> List[str]:
    # Obtém features combinadas de forma dinâmica
    client_features = get_client_features(user_id, clients_features_df).to_frame().T
    non_viewed_news = get_non_viewed_news(user_id, news_features_df)

    # Para cada noticia combina com as features do usuario
    model_input = non_viewed_news.assign(user_id=user_id).merge(
        client_features.drop(columns=["user_id"]), how="cross", suffixes=("_news", "_user")
    )

    scores = model.predict(model_input)
    return get_predicted_news(scores, non_viewed_news, n=n, score_threshold=score_threshold)
