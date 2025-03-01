"""
Módulo com funções para realizar predições com o modelo de recomendação.
"""
import os
import pandas as pd
from typing import List, Optional

from config import logger, DATA_PATH, USE_S3
from data.data_loader import get_client_features, get_non_viewed_news, get_predicted_news
from storage.io import Storage
from .constants import CLIENT_FEATURES, NEWS_FEATURES


def prepare_for_prediction(storage: Optional[Storage] = None) -> None:
    """
    Prepara os dados para predição, separando features de notícias e clientes.

    Args:
        storage (Storage, opcional): Instância de Storage para I/O.
            Se não for fornecido, cria uma nova instância.
    """
    # Se não for fornecido um objeto Storage, cria um novo
    if storage is None:
        storage = Storage(use_s3=USE_S3)

    logger.info("Utilizando armazenamento %s.", "remoto" if USE_S3 else "local")

    # Caminho completo para o arquivo de features
    file_path = os.path.join(DATA_PATH, "train", "X_train.parquet")
    logger.info("Lendo arquivo de features: %s", file_path)

    # Carrega o DataFrame de features
    df = storage.read_parquet(file_path)

    # Separa o DataFrame em dois: news_features_df e clients_features_df
    news_features_df = df[['historyId', 'pageId'] + NEWS_FEATURES].copy()
    clients_features_df = df[['userId', 'pageId'] + CLIENT_FEATURES].copy()

    # Define o diretório de saída para as predições
    predict_path = os.path.join(DATA_PATH, "predict")

    # Caminhos completos para salvar os arquivos
    news_save_path = os.path.join(predict_path, "news_features_df.parquet")
    clients_save_path = os.path.join(predict_path, "clients_features_df.parquet")

    # Salva os DataFrames em formato Parquet
    storage.write_parquet(news_features_df, news_save_path)
    storage.write_parquet(clients_features_df, clients_save_path)
    logger.info("Arquivos salvos: %s e %s", news_save_path, clients_save_path)


def predict_for_userId(
    userId: str,
    news_features_df: pd.DataFrame,
    clients_features_df: pd.DataFrame,
    model,
    n: int = 5,
    score_threshold: float = 0.3,
) -> List[str]:
    """
    Gera recomendações de notícias para um usuário específico.

    Args:
        userId (str): ID do usuário.
        news_features_df (pd.DataFrame): DataFrame com as features das notícias.
        clients_features_df (pd.DataFrame): DataFrame com as features dos clientes.
        model: Modelo treinado para predição.
        n (int, opcional): Número máximo de recomendações. Padrão: 5.
        score_threshold (float, opcional): Limite mínimo de score. Padrão: 0.3.

    Returns:
        List[str]: Lista de IDs das notícias recomendadas.
    """
    try:
        # Obtém as features do cliente e converte para DataFrame
        # (já que get_client_features retorna um dicionário)
        client_features = pd.DataFrame([get_client_features(userId, clients_features_df)])

        # Obtém as notícias que o usuário ainda não visualizou
        non_viewed_news = get_non_viewed_news(userId, news_features_df, clients_features_df)

        # Se não houver notícias disponíveis, retorna uma lista vazia
        if non_viewed_news.empty:
            logger.warning("Nenhuma notícia disponível para recomendar.")
            return []

        # Cria o input do modelo: combina as features das notícias com as features do usuário
        model_input = non_viewed_news.assign(userId=userId).merge(
            client_features.drop(columns=["userId"]), how="cross", suffixes=("_news", "_user")
        )

        # Calcula os scores usando o modelo
        scores = model.predict(model_input)

        # Retorna as notícias recomendadas aplicando o filtro de score
        return get_predicted_news(scores, non_viewed_news, n=n, score_threshold=score_threshold)

    except Exception as e:
        logger.error(f"Erro ao gerar recomendações para o usuário {userId}: {e}")
        return []
