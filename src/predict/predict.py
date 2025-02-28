#!/usr/bin/env python
import os
import sys
import pandas as pd
from typing import List

from data.data_loader import get_client_features, get_non_viewed_news, get_predicted_news
from config import FLAG_REMOTE, LOCAL_DATA_PATH, REMOTE_DATA_PATH, logger
from constants import CLIENT_FEATURES, NEWS_FEATURES


def prepare_for_prediction():
    # Define o caminho base dos dados conforme o ambiente
    data_path = REMOTE_DATA_PATH if FLAG_REMOTE else LOCAL_DATA_PATH
    logger.info("Utilizando armazenamento %s.", "remoto" if FLAG_REMOTE else "local")
    
    # Caminho completo para o arquivo de features
    file_path = os.path.join(data_path, "train", "X_train.parquet")
    logger.info("Lendo arquivo de features: %s", file_path)
    
    # Carrega o DataFrame de features
    df = pd.read_parquet(file_path)
    
    # TODO: obter novamente as chaves
    # Separa o DataFrame em dois: news_features_df e clients_features_df
    news_features_df = df[['historyId', 'pageId'] + NEWS_FEATURES].copy()
    clients_features_df = df[['userId', 'pageId'] + CLIENT_FEATURES].copy()
    
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
    
    return news_save_path, clients_save_path


def predict_for_userId(
    userId: str,
    news_features_df: pd.DataFrame,
    clients_features_df: pd.DataFrame,
    model,
    n: int = 5,
    score_threshold: float = 0.3,
) -> List[str]:
    # Obtém as features do cliente e converte para DataFrame (já que get_client_features retorna um dicionário)
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


def main():
    # Primeiro, prepara os dados para predição
    logger.info("Iniciando preparação dos dados para predição...")
    news_save_path, clients_save_path = prepare_for_prediction()
    
    # Define o caminho base dos dados conforme o ambiente
    data_path = REMOTE_DATA_PATH if FLAG_REMOTE else LOCAL_DATA_PATH
    predict_path = os.path.join(data_path, "predict")
    
    # Carrega os DataFrames preparados
    news_features_df = pd.read_parquet(os.path.join(predict_path, "news_features_df.parquet"))
    clients_features_df = pd.read_parquet(os.path.join(predict_path, "clients_features_df.parquet"))
    
    # Carregamento do modelo
    # Substitua a seguir pela sua lógica de carregamento do modelo treinado.
    # Exemplo: 
    # from my_model_module import load_model
    # model = load_model()
    #
    # Para exemplificar, usamos um modelo dummy:
    class DummyModel:
        def predict(self, X):
            # Retorna um score fixo para todos os exemplos
            return [0.5] * len(X)
    model = DummyModel()
    
    # Solicita o userId para predição
    userId = input("Informe o userId para predição: ").strip()
    if not userId:
        logger.error("UserId não informado. Encerrando.")
        sys.exit(1)
    
    # Realiza a predição
    logger.info("Realizando predição para o userId: %s", userId)
    recommendations = predict_for_userId(userId, news_features_df, clients_features_df, model)
    
    # Exibe as recomendações
    print("Recomendações para o usuário {}:".format(userId))
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        print("Nenhuma recomendação gerada.")


if __name__ == '__main__':
    main()
