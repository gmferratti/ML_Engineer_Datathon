import os
import pandas as pd
from typing import List, Optional
from src.config import DATA_PATH, logger, USE_S3
from storage.io import Storage
from src.data.data_loader import get_client_features, get_non_viewed_news
# Note: IMPORT get_predicted_news se necessário
from src.predict.constants import EXPECTED_COLUMNS


def prepare_for_prediction(storage: Optional[Storage] = None) -> None:
    """
    Prepara os dados para predição separando features de notícias e clientes.

    Args:
        storage (Storage, optional): Instância para I/O. Cria se None.
    """
    if storage is None:
        storage = Storage(use_s3=USE_S3)
    logger.info("Utilizando armazenamento: %s", "S3" if USE_S3 else "local")
    file_path = os.path.join(DATA_PATH, "train", "X_train.parquet")
    logger.info("Lendo features de: %s", file_path)
    df = storage.read_parquet(file_path)
    news_df = df[["historyId", "pageId"] + EXPECTED_COLUMNS].copy()
    client_df = df[["userId", "pageId"] + EXPECTED_COLUMNS].copy()
    pred_dir = os.path.join(DATA_PATH, "predict")
    news_save = os.path.join(pred_dir, "news_features_df.parquet")
    client_save = os.path.join(pred_dir, "clients_features_df.parquet")
    storage.write_parquet(news_df, news_save)
    storage.write_parquet(client_df, client_save)
    logger.info("Arquivos salvos: %s e %s", news_save, client_save)


def predict_for_userId(userId: str, news_df: pd.DataFrame,
                       client_df: pd.DataFrame, model,
                       n: int = 5, score_threshold: float = 0.3) -> List[str]:
    """
    Gera recomendações de notícias para um usuário.

    Args:
        userId (str): ID do usuário.
        news_df (pd.DataFrame): Features das notícias.
        client_df (pd.DataFrame): Features dos clientes.
        model: Modelo treinado.
        n (int, optional): Máximo de recomendações. Default: 5.
        score_threshold (float, optional): Score mínimo. Default: 0.3.

    Returns:
        List[str]: Lista de IDs recomendados.
    """
    client_feat = pd.DataFrame([get_client_features(userId, client_df)])
    non_viewed = get_non_viewed_news(userId, news_df, client_df)
    if non_viewed.empty:
        logger.warning("Nenhuma notícia disponível para recomendar.")
        return []
    model_input = non_viewed.assign(userId=userId).merge(
        client_feat.drop(columns=["userId"]), how="cross",
        suffixes=("_news", "_user")
    )
    scores = model.predict(model_input)
    from src.data.data_loader import get_predicted_news
    return get_predicted_news(scores, non_viewed, n=n,
                              score_threshold=score_threshold)
