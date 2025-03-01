# predict.py
import os
import sys
import pandas as pd
from typing import List
import mlflow

from data.data_loader import get_client_features, get_non_viewed_news, get_predicted_news
from config import FLAG_REMOTE, LOCAL_DATA_PATH, REMOTE_DATA_PATH, logger, get_config
from constants import EXPECTED_COLUMNS
#TODO: Importar load_mlflow. Evitar redundância.

def prepare_for_prediction() -> str:
    """
    Lê o dataframe completo de features e o salva no diretório de predição.
    Retorna o caminho do arquivo salvo.
    """
    # Define o caminho base dos dados conforme o ambiente
    data_path = REMOTE_DATA_PATH if FLAG_REMOTE else LOCAL_DATA_PATH
    logger.info("Utilizando armazenamento %s.", "remoto" if FLAG_REMOTE else "local")
    
    # Caminho completo para o arquivo de features completo
    X_train_full_path = os.path.join(data_path, "train", "X_train_full.parquet")
    logger.info("Lendo arquivo de features: %s", X_train_full_path)
    
    # Carrega o dataframe completo
    X_train_full = pd.read_parquet(X_train_full_path)
    
    # Define o diretório de saída para as predições e salva o dataframe completo
    predict_path = os.path.join(data_path, "predict")
    os.makedirs(predict_path, exist_ok=True)
    logger.info("Salvando arquivo de predição na pasta: %s", predict_path)
    
    full_save_path = os.path.join(predict_path, "X_train_full.parquet")
    X_train_full.to_parquet(full_save_path)
    logger.info("Arquivo salvo: %s", full_save_path)
    
    return full_save_path



def predict_for_userId(
    userId: str,
    full_df: pd.DataFrame,
    model,
    n: int = 5,
    score_threshold: int = 15,
) -> List[str]:
    """
    Gera recomendações para um dado userId utilizando o dataframe completo.
    Essa implementação assume que as features de notícias (com sufixo _news)
    são as que devem ser utilizadas para a predição.
    """
    # Filtra as notícias que o usuário já viu (usando a coluna 'pageId')
    seen_pages = full_df.loc[full_df['userId'] == userId, 'pageId'].unique()
    non_viewed_news = full_df[~full_df['pageId'].isin(seen_pages)].copy()
    
    if non_viewed_news.empty:
        logger.warning("Nenhuma notícia disponível para recomendar.")
        return []
    
    # Obtém as features do usuário a partir do dataframe completo
    client_features = full_df.loc[full_df['userId'] == userId].iloc[0]
    client_features_df = pd.DataFrame([client_features])
    
    #TODO:Lógica aqui
    
    # Calcula os scores usando o modelo
    scores = model.predict(model_input)

    # Retorna as notícias recomendadas aplicando o filtro de score
    return get_predicted_news(scores, non_viewed_news, n=n, score_threshold=score_threshold)


def main():
    # Prepara os dados para predição (salva o dataframe completo)
    logger.info("Iniciando preparação dos dados para predição...")
    full_save_path = prepare_for_prediction()
    
    data_path = REMOTE_DATA_PATH if FLAG_REMOTE else LOCAL_DATA_PATH
    predict_path = os.path.join(data_path, "predict")
    
    # Carrega o dataframe completo de predição
    full_df = pd.read_parquet(os.path.join(predict_path, "X_train_full.parquet"))
    
    # Carrega o modelo via MLflow
    model = load_mlflow_model()
    
    # userId random
    userId = "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297"

    # Realiza a predição
    logger.info("Realizando predição para o userId: %s", userId)
    recommendations = predict_for_userId(userId, full_df, model)
    
    # Exibe as recomendações
    logger.info("Recomendações para o usuário {}:".format(userId))
    if recommendations:
        for rec in recommendations:
            print(rec)
    else:
        logger.info("Nenhuma recomendação gerada.")


if __name__ == '__main__':
    main()