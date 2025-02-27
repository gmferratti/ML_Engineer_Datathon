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
      3. Salva todos os dados contidos em trusted_data em formato Parquet,
         utilizando um loop para evitar hardcoding.
      4. Chama load_train_data para validar o carregamento dos dados.
    """
    # 1. Carregar o DataFrame final com features e target
    final_feats_path = os.path.join("data", "features", "final_feats_with_target.parquet")
    logger.info("Carregando features finais com target de %s...", final_feats_path)
    final_feats_with_target = pd.read_parquet(final_feats_path)

    # 2. Preparar os conjuntos de treino e teste
    logger.info("Preparando features...")
    trusted_data = prepare_features(final_feats_with_target)

    base_train_path = os.path.join("data", "train")
    os.makedirs(base_train_path, exist_ok=True)

    # 3. Salvar todos os itens de trusted_data em formato Parquet utilizando loop
    for key, data in trusted_data.items():
        # Se for o encoder_mapping e for um dict, converte para DataFrame
        if key == "encoder_mapping" and isinstance(data, dict):
            data = pd.DataFrame(data)
        # Se não for um DataFrame, tenta convertê-lo (a maioria dos itens devem ser DataFrames)
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as e:
                logger.warning("Não foi possível converter %s para DataFrame: %s", key, e)
                continue

        file_path = os.path.join(base_train_path, f"{key}.parquet")
        logger.info("Salvando %s em %s", key, file_path)
        _save_df_parquet(data, file_path)

    logger.info("Pipeline de treinamento concluído!")

    # 4. Validar o carregamento dos dados
    X_loaded, y_loaded = load_train_data()
    logger.info("Dados carregados: X_train shape: %s, y_train shape: %s", X_loaded.shape, y_loaded.shape)

if __name__ == "__main__":
    train_model()
