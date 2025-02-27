import os
import pandas as pd
from typing import List, Dict
from config import logger

def get_client_features(
    userId: str,
    clients_features_df: pd.DataFrame
) -> Dict:
    return (
        clients_features_df[
            clients_features_df["userId"] == userId
        ].to_dict(orient="records")
    )[0]


def get_non_viewed_news(
    userId: str,
    news_features_df: pd.DataFrame,
    clients_features_df: pd.DataFrame
) -> pd.DataFrame:
    """Pega as noticias que o usuario ainda nao viu.

    Args:
        userId (str): ID do usuario.
        news_features_df (DataFrame): DataFrame com as features das noticias.
        clients_features_df (DataFrame): DataFrame com o histórico completo dos usuários.

    Returns:
        DataFrame: DataFrame com as noticias que o usuario ainda nao viu.
    """
    # Recupera as páginas (pageId) já visualizadas pelo usuário
    read_pages = clients_features_df.loc[clients_features_df['userId'] == userId, 'pageId'].unique()
    
    # Filtra as notícias que não foram visualizadas
    unread = news_features_df[~news_features_df['pageId'].isin(read_pages)].copy()
    
    # Adiciona a coluna 'userId' para identificar a qual usuário as notícias se referem
    unread['userId'] = userId
    
    # Seleciona e organiza as colunas de interesse
    unread = unread[['userId', 'pageId']].reset_index(drop=True)
    
    return unread

def get_predicted_news(
    scores: List[float],
    news_features_df: pd.DataFrame,
    n: int = 5,
    score_threshold: float = 0.3,
) -> List[str]:
    """Pega as noticias recomendadas.

    Args:
        scores (List[float]): Scores das noticias.
        news_features_df (DataFrame): DataFrame com as features das noticias.
        n (int): Quantidade de noticias a recomendar (default: 5).
        score_threshold (float): Score minimo para considerar a recomendacao.

    Returns:
        List[str]: Lista de IDs das noticias recomendadas.
    """
    # TODO: Implementar logica
    return news_features_df.head(n)["historyId"].tolist()


def get_evaluation_data() -> pd.DataFrame:
    """Pega os dados de avaliação.

    Returns:
        pd.DataFrame: DataFrame com os dados de avaliação (features + target)
    """
    # Caminhos dos arquivos de avaliação
    X_test_path = "data/train/X_test.parquet"
    y_test_path = "data/train/y_test.parquet"

    # Carregar os DataFrames
    X_test = pd.read_parquet(X_test_path)
    y_test = pd.read_parquet(y_test_path)
    
    # Adiciona a coluna TARGET ao DataFrame de features
    X_test["TARGET"] = y_test
    
    # Renomeia DataFrame de features
    eval_df = X_test

    return eval_df
