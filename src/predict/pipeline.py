import os
import pandas as pd
import pickle
from config import FLAG_REMOTE, LOCAL_DATA_PATH, REMOTE_DATA_PATH, logger
from predict import predict_for_userId

def prediction():
    # Define o caminho base conforme o ambiente: remoto ou local
    data_path = REMOTE_DATA_PATH if FLAG_REMOTE else LOCAL_DATA_PATH
    logger.info("Utilizando armazenamento %s.", "remoto" if FLAG_REMOTE else "local")
    
    # Diretório onde os DataFrames salvos estão armazenados
    predict_dir = os.path.join(data_path, "predict")
    
    # Caminhos completos para os arquivos de features
    news_features_file = os.path.join(predict_dir, "news_features_df.parquet")
    clients_features_file = os.path.join(predict_dir, "clients_features_df.parquet")
    
    logger.info("Lendo arquivos de features: %s e %s", news_features_file, clients_features_file)
    
    # Carrega os DataFrames salvos
    news_features_df = pd.read_parquet(news_features_file)
    clients_features_df = pd.read_parquet(clients_features_file)
    
    # Caminho do modelo salvo (lightgbm_ranker.pkl) na pasta train
    model_file = os.path.join(data_path, "train", "lightgbm_ranker.pkl")
    logger.info("Carregando modelo salvo: %s", model_file)
    
    # Carrega o modelo treinado (supondo que foi salvo com pickle)
    with open(model_file, "rb") as f:
        model = pickle.load(f)
    
    # Testa a função de predição para um usuário com histórico
    regular_user_hash = "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297"
    recommendations_existing = predict_for_userId(regular_user_hash, news_features_df, clients_features_df, model)
    print(f"Recomendações para usuário '{regular_user_hash}':", recommendations_existing)
