import os
import pandas as pd
from typing import List, Dict, Optional
from src.config import logger, DATA_PATH, USE_S3
from storage.io import Storage
from src.predict.constants import CLIENT_FEATURES_COLUMNS, NEWS_FEATURES_COLUMNS

def get_client_features(userId: str, clients_features_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Obtém as características de um cliente.

    Args:
        userId (str): ID do usuário.
        clients_features_df (pd.DataFrame): Dados dos clientes.

    Returns:
        pd.Series or None: Características do cliente, ou None se não encontrado.
    """
    df = clients_features_df[clients_features_df["userId"] == userId]
    if df.empty:
        logger.warning("Nenhuma feature encontrada para o usuário: %s", userId)
        return None
    return df.iloc[0]


def get_non_viewed_news(userId: str, news_features_df: pd.DataFrame,
                        clients_features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna notícias que o usuário ainda não visualizou.

    Args:
        userId (str): ID do usuário.
        news_features_df (pd.DataFrame): Dados das notícias.
        clients_features_df (pd.DataFrame): Histórico dos usuários.

    Returns:
        pd.DataFrame: Notícias não visualizadas.
    """
    seen = clients_features_df.loc[
        clients_features_df["userId"] == userId, "pageId"
    ].unique()
    unread = news_features_df[~news_features_df["pageId"].isin(seen)].copy()
    unread["userId"] = userId
    return unread[["userId", "pageId"]].reset_index(drop=True)


def get_predicted_news(scores: List[float],
                       news_features_df: pd.DataFrame,
                       n: int = 5,
                       score_threshold: float = 30) -> List[str]:
    """
    Retorna os IDs das notícias recomendadas com base nos scores.

    Args:
        scores (List[float]): Scores previstos.
        news_features_df (pd.DataFrame): Dados das notícias.
        n (int, optional): Máximo de notícias. Default: 5.
        score_threshold (float, optional): Score mínimo. Default: 30.

    Returns:
        List[str]: IDs das notícias recomendadas.
    """
    df_scores = pd.DataFrame({
        "pageId": news_features_df["pageId"],
        "score": scores
    })
    filtered = df_scores[df_scores["score"] >= score_threshold]
    top_news = filtered.sort_values("score", ascending=False).head(n)
    return top_news["pageId"].tolist()


def get_evaluation_data(storage: Optional[Storage] = None) -> pd.DataFrame:
    """
    Carrega dados de avaliação (features + target).

    Args:
        storage (Storage, optional): Instância para I/O.

    Returns:
        pd.DataFrame: Dados de avaliação.
    """
    if storage is None:
        storage = Storage(use_s3=USE_S3)
    X_path = os.path.join(DATA_PATH, "train", "X_test.parquet")
    y_path = os.path.join(DATA_PATH, "train", "y_test.parquet")
    X_test = storage.read_parquet(X_path)
    y_test = storage.read_parquet(y_path)
    X_test["TARGET"] = y_test
    return X_test


def load_data_for_prediction(storage: Optional[Storage] = None) -> Dict[str, pd.DataFrame]:
    """
    Carrega dados para predição (notícias e clientes) a partir do arquivo completo de features,
    separando em dois DataFrames: um para notícias e outro para clientes.
    """
    if storage is None:
        storage = Storage(use_s3=USE_S3)
    
    full_path = os.path.join(DATA_PATH, "train", "X_train_full.parquet")
    logger.info("🔍 [Data Loader] Carregando dados completos de: %s", full_path)
    full_df = storage.read_parquet(full_path)
    
    # Inclua a coluna 'pageId' no DataFrame de notícias
    if 'pageId' not in full_df.columns:
        logger.error("🚨 [Data Loader] A coluna 'pageId' não foi encontrada no DataFrame completo.")
        raise KeyError("Coluna 'pageId' ausente no arquivo completo de features.")
    
    news_features_df = full_df[['pageId'] + NEWS_FEATURES_COLUMNS]
    
    # Extrai as features dos clientes: 'userId' + demais features, removendo duplicatas
    if 'userId' not in full_df.columns:
        logger.error("🚨 [Data Loader] A coluna 'userId' não foi encontrada no DataFrame completo.")
        raise KeyError("Coluna 'userId' ausente.")
    clients_features_df = full_df[['userId'] + CLIENT_FEATURES_COLUMNS].drop_duplicates()
    
    logger.info("✅ [Data Loader] Dados preparados: %d registros de notícias e %d de clientes.",
                len(news_features_df), len(clients_features_df))
    
    return {"news_features": news_features_df, "clients_features": clients_features_df}


def load_model(storage: Optional[Storage] = None):
    """
    Carrega o modelo treinado para predição.

    Args:
        storage (Storage, optional): Instância para I/O.

    Returns:
        object: Modelo treinado ou None.
    """
    if storage is None:
        storage = Storage(use_s3=USE_S3)
    model_path = os.path.join(DATA_PATH, "train", "lightgbm_ranker.pkl")
    try:
        model = storage.load_pickle(model_path)
        logger.info("Modelo carregado de %s", model_path)
        return model
    except Exception as e:
        logger.error("Erro ao carregar modelo de %s: %s", model_path, e)
        return None
