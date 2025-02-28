import pandas as pd
from typing import List


def get_client_features(
    user_id: str,
    clients_features_df: pd.DataFrame
) -> pd.Series:
    return clients_features_df[clients_features_df["user_id"] == user_id].iloc[0]


def get_non_viewed_news(
    user_id: str,
    news_features_df: pd.DataFrame
) -> pd.DataFrame:
    # TODO: Implementar lógica real de filtragem
    return news_features_df


def get_predicted_news(
    scores: List[float],
    news_features_df: pd.DataFrame,
    n: int = 5,
    score_threshold: float = 0.3,
) -> List[str]:
    # Exemplo: retorna os primeiros 'n' IDs usando a coluna "news_id"
    return news_features_df.head(n)["news_id"].tolist()


def get_evaluation_data() -> pd.DataFrame:
    # TODO: Implementar lógica de carregamento dos dados de avaliação
    return pd.DataFrame()
