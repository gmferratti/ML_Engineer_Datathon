import os
import pandas as pd
from typing import List, Dict, Optional, Any
from src.config import logger, DATA_PATH, USE_S3
from src.storage.io import Storage
from src.predict.constants import CLIENT_FEATURES_COLUMNS, NEWS_FEATURES_COLUMNS, METADATA_COLS


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
                       score_threshold: float = 30) -> List[Dict[str, Any]]:
    """
    Retorna os IDs e os scores das notícias recomendadas com base nos scores.

    Args:
        scores (List[float]): Scores previstos.
        news_features_df (pd.DataFrame): Dados das notícias.
        n (int, opcional): Máximo de notícias. Default: 5.
        score_threshold (float, opcional): Score mínimo. Default: 30.

    Returns:
        List[Dict[str, Any]]: Lista de dicionários com 'pageId' e 'score'.
    """
    df_scores = pd.DataFrame({
        "pageId": news_features_df["pageId"],
        "score": scores
    })
    filtered = df_scores[df_scores["score"] >= score_threshold]
    top_news = filtered.sort_values("score", ascending=False).head(n)
    return top_news.to_dict("records")


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


def load_data_for_prediction(storage: Optional[Storage] = None,
                             include_metadata: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Carrega os dados para predição a partir do arquivo completo de features,
    separando-os em DataFrames de notícias e clientes. Opcionalmente, realiza o merge
    com metadados das notícias (como 'title' e 'url').

    Args:
        storage (Optional[Storage]): Instância de armazenamento para I/O. Se None,
            uma nova instância será criada.
        include_metadata (bool): Flag para indicar se os metadados das notícias devem
            ser incluídos. Padrão é False.

    Returns:
        Dict[str, pd.DataFrame]: Dicionário contendo:
            - "news_features": DataFrame com as features das notícias
             (e metadados, se solicitado).
            - "clients_features": DataFrame com as features dos clientes.
    """
    if storage is None:
        storage = Storage(use_s3=USE_S3)

    full_path = os.path.join(DATA_PATH, "train", "X_train_full.parquet")
    logger.info("🔍 [Data Loader] Carregando dados completos de: %s", full_path)
    full_df = storage.read_parquet(full_path)

    # Extrai as features de notícias, garantindo que 'pageId' esteja incluído
    news_features_df = full_df[['pageId'] + NEWS_FEATURES_COLUMNS]

    # Converte pageId para string para garantir a compatibilidade
    news_features_df.loc[:, "pageId"] = news_features_df["pageId"].astype(str)

    if include_metadata:
        try:
            news_metadata = pd.read_parquet("data/features/news_feats.parquet")[METADATA_COLS]
            # Converte também os pageId dos metadados
            news_metadata["pageId"] = news_metadata["pageId"].astype(str)
            logger.info("🔍 [Data Loader] Metadados de notícias carregados com sucesso.")
            news_features_df = news_features_df.merge(news_metadata, on="pageId", how="left")
            logger.info("✅ [Data Loader] DataFrame de notícias enriquecido: %d registros.",
                        len(news_features_df))
        except Exception as e:
            logger.warning(
                "Não foi possível carregar ou fazer merge dos metadados das notícias: %s", e)

    if 'userId' not in full_df.columns:
        logger.error(
            "🚨 [Data Loader] A coluna 'userId' não foi encontrada no DataFrame completo.")
        raise KeyError("Coluna 'userId' ausente.")
    clients_features_df = full_df[['userId'] + CLIENT_FEATURES_COLUMNS].drop_duplicates()

    logger.info("✅ [Data Loader] Dados preparados: %d registros de notícias e %d de clientes.",
                len(news_features_df), len(clients_features_df))

    return {"news_features": news_features_df, "clients_features": clients_features_df}
