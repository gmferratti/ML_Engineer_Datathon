import os
import pandas as pd

from data.data_loader import get_client_features, get_non_viewed_news, get_predicted_news
from typing import List

from config import FLAG_REMOTE, LOCAL_DATA_PATH, REMOTE_DATA_PATH, logger
from .constants import CLIENT_FEATURES, NEWS_FEATURES


def prepare_for_prediction():
    # Define o caminho base dos dados conforme o ambiente
    data_path = REMOTE_DATA_PATH if FLAG_REMOTE else LOCAL_DATA_PATH
    logger.info("Utilizando armazenamento %s.", "remoto" if FLAG_REMOTE else "local")

    # Caminho completo para o arquivo de features
    file_path = os.path.join(data_path, "train", "X_train.parquet")
    logger.info("Lendo arquivo de features: %s", file_path)

    # Carrega o DataFrame de features
    df = pd.read_parquet(file_path)

    # Separa o DataFrame em dois: news_features_df e clients_features_df
    news_features_df = df[["historyId", "pageId"] + NEWS_FEATURES].copy()
    clients_features_df = df[["userId", "pageId"] + CLIENT_FEATURES].copy()

    # Define o diretório de saída para as predições
    predict_path = os.path.join(data_path, "predict")
    os.makedirs(predict_path, exist_ok=True)
    logger.info("Salvando arquivos na pasta: %s", predict_path)

    # Caminhos completos para salvar os arquivos
    news_save_path = os.path.join(predict_path, "news_features_df.parquet")
    clients_save_path = os.path.join(predict_path, "clients_features_df.parquet")

    # Salva os DataFrames em formato Parquet
    news_features_df.to_parquet(news_save_path)
    clients_features_df.to_parquet(clients_save_path)
    logger.info("Arquivos salvos: %s e %s", news_save_path, clients_save_path)


def predict_for_userId(
    userId: str,
    news_features_df: pd.DataFrame,
    clients_features_df: pd.DataFrame,
    model,
    n: int = 5,
    score_threshold: float = 0.3,
) -> List[str]:
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
