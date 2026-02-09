import os
from typing import Any, Dict, List, Optional

import pandas as pd

from src.config import DATA_PATH, USE_S3, logger
from src.predict.constants import (
    CLIENT_FEATURES_COLUMNS, 
    METADATA_COLS, 
    NEWS_FEATURES_COLUMNS)


def get_client_features(user_id: str, clients_df: pd.DataFrame) -> Optional[pd.Series]:
    """
    Retorna as features do cliente identificado por `user_id`.

    Args:
        user_id: Identificador do usuário.
        clients_df: DataFrame contendo pelo menos a coluna `userId` e as
            colunas listadas em `CLIENT_FEATURES_COLUMNS`.

    Returns:
        Série com as colunas do cliente, ou `None` se não houver registro.
    """
    matches = clients_df[clients_df["userId"] == user_id]
    if matches.empty:
        logger.warning("Nenhuma feature encontrada para o usuário: %s", user_id)
        return None
    return matches.iloc[0]


def get_non_viewed_news(user_id: str, news_df: pd.DataFrame, clients_df: pd.DataFrame) -> pd.DataFrame:
    """
    Retorna as notícias que o usuário ainda não visualizou.

    Args:
        user_id: Identificador do usuário.
        news_df: DataFrame de notícias contendo a coluna `pageId`.
        clients_df: Histórico de visualizações contendo colunas `userId` e `pageId`.

    Returns:
        DataFrame com colunas `userId` e `pageId` para itens não vistos.
    """
    seen_page_ids = clients_df.loc[clients_df["userId"] == user_id, "pageId"].unique()
    not_seen = news_df[~news_df["pageId"].isin(seen_page_ids)].copy()
    not_seen = not_seen.reset_index(drop=True)
    not_seen["userId"] = user_id
    return not_seen[["userId", "pageId"]]


def get_predicted_news(
    scores: List[float], news_df: pd.DataFrame, n: int = 5, score_threshold: float = 30.0
) -> List[Dict[str, Any]]:
    """
    Seleciona as notícias previstas com maior score.

    Args:
        scores: Lista de scores, alinhada por posição com `news_df`.
        news_df: DataFrame contendo a coluna `pageId`.
        n: Número máximo de itens retornados.
        score_threshold: Valor mínimo de score para incluir a notícia.

    Returns:
        Lista de dicionários com chaves `pageId` e `score`, ordenada por score decrescente.
    """
    scores_df = pd.DataFrame({"pageId": news_df["pageId"].astype(str).values, "score": scores})
    selected = scores_df[scores_df["score"] >= score_threshold]
    top_n = selected.sort_values("score", ascending=False).head(n)
    return top_n.to_dict("records")


def get_evaluation_data(storage: Optional[object] = None) -> pd.DataFrame:
    """
    Carrega os dados de avaliação combinando features e target.

    Args:
        storage: Instância de storage com método `read_parquet(path)`.

    Returns:
        DataFrame com colunas de features e coluna `TARGET` contendo o vetor alvo.
    """
    if storage is None:
        from src.storage.io import Storage as _Storage

        storage = _Storage(use_s3=USE_S3)
    x_path = os.path.join(DATA_PATH, "train", "X_test.parquet")
    y_path = os.path.join(DATA_PATH, "train", "y_test.parquet")
    x_df = storage.read_parquet(x_path)
    y_series = storage.read_parquet(y_path)
    x_df = x_df.copy()
    x_df["TARGET"] = y_series
    return x_df


def load_data_for_prediction(storage: Optional[object] = None, include_metadata: bool = False) -> Dict[str, pd.DataFrame]:
    """
    Carrega o DataFrame completo de features e separa em duas estruturas:
    - `news_features`: features por `pageId` (opcionalmente com metadados);
    - `clients_features`: features por `userId`.

    Args:
        storage: Instância de storage (usa Storage local por padrão).
        include_metadata: Se True, tenta enriquecer `news_features` com metadados.

    Returns:
        Dicionário com as chaves `news_features` e `clients_features`.
    """
    if storage is None:
        from src.storage.io import Storage as _Storage

        storage = _Storage(use_s3=USE_S3)
    full_path = os.path.join(DATA_PATH, "train", "X_train_full.parquet")
    logger.info("[Data Loader] Carregando dados completos de: %s", full_path)
    full_df = storage.read_parquet(full_path)

    news_df = full_df[["pageId"] + NEWS_FEATURES_COLUMNS].copy()
    news_df["pageId"] = news_df["pageId"].astype(str)

    if include_metadata:
        try:
            metadata_df = pd.read_parquet(os.path.join("data", "features", "news_feats.parquet"))[METADATA_COLS]
            metadata_df["pageId"] = metadata_df["pageId"].astype(str)
            news_df = news_df.merge(metadata_df, on="pageId", how="left")
            logger.info("[Data Loader] Metadados de notícias adicionados: %d registros.", len(news_df))
        except Exception as exc:  # pragma: no cover - IO environment dependent
            logger.warning("Falha ao carregar metadados das notícias: %s", exc)

    if "userId" not in full_df.columns:
        logger.error("Coluna 'userId' não encontrada no DataFrame completo.")
        raise KeyError("Coluna 'userId' ausente no dataset completo.")

    clients_df = full_df[["userId"] + CLIENT_FEATURES_COLUMNS].drop_duplicates().reset_index(drop=True)
    logger.info("[Data Loader] Dados preparados: %d notícias, %d clientes.", len(news_df), len(clients_df))

    return {"news_features": news_df, "clients_features": clients_df}
