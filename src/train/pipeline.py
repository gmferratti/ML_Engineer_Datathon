"""Wrapper para o pipeline de treinamento do modelo."""

import pandas as pd
import os
from utils import prepare_features, load_train_data
from config import logger

def _save_df_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Cria o diretório (caso não exista) e salva o DataFrame em formato parquet.

    Args:
        df (pandas.DataFrame): DataFrame a ser salvo.
        file_path (str): Caminho completo para salvar o arquivo parquet.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    df.to_parquet(file_path)
    logger.info(f"Arquivo salvo em {file_path}")

def train_model() -> None:
    """
    Pipeline de treinamento do modelo:
      1. Carrega o DataFrame final com features e target.
      2. Prepara os conjuntos de treino e teste.
      3. Salva os dados em formato parquet.
      4. Chama load_train_data para validar o carregamento dos dados.
    """
    # 1. Carregar o DataFrame final com features e target
    final_feats_path = os.path.join("data", "features", "final_feats_with_target.parquet")
    logger.info(f"Carregando features finais com target de {final_feats_path}...")
    final_feats_with_target = pd.read_parquet(final_feats_path)
    
    # 2. Preparar os conjuntos de treino e teste
    logger.info("Preparando features...")
    trusted_data = prepare_features(final_feats_with_target)
    
    X_train = trusted_data["X_train"]
    X_test = trusted_data["X_test"]
    y_train = pd.DataFrame(trusted_data["y_train"])
    y_test = pd.DataFrame(trusted_data["y_test"])
    
    base_train_path = os.path.join("data", "train")
    
    # 3. Salvar os dados de treino e teste em formato parquet
    logger.info("Salvando conjuntos de treino e target em formato parquet...")
    _save_df_parquet(X_train, os.path.join(base_train_path, "X_train.parquet"))
    _save_df_parquet(X_test, os.path.join(base_train_path, "X_test.parquet"))
    _save_df_parquet(y_train, os.path.join(base_train_path, "y_train.parquet"))
    _save_df_parquet(y_test, os.path.join(base_train_path, "y_test.parquet"))
    
    encoder_mapping = pd.DataFrame(trusted_data["encoder_mapping"])
    _save_df_parquet(encoder_mapping, os.path.join(base_train_path, "encoder_mapping.parquet"))
    
    logger.info("Pipeline de treinamento concluído!")
    
    # 4. Chamada de load_train_data para validar o carregamento dos dados
    X_loaded, y_loaded = load_train_data()
    logger.info(f"Dados carregados: X_train shape: {X_loaded.shape}, y_train shape: {y_loaded.shape}")
    
    # TODO: Só treinar com quem não for cold_start

if __name__ == "__main__":
    train_model()
