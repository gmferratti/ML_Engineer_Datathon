"""
Wrapper para o pipeline de treinamento do modelo.
"""

import os
import pandas as pd
from utils import prepare_features, load_train_data
from config import logger


def save_dataframe_as_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Cria o diretório (caso não exista) e salva o DataFrame em formato Parquet.

    Args:
        df (pd.DataFrame): DataFrame a ser salvo.
        file_path (str): Caminho completo para salvar o arquivo Parquet.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path)
    # Exibe apenas o caminho relativo, pois o diretório base já foi informado no início.
    rel_path = os.path.relpath(file_path)
    logger.info("Arquivo salvo: %s", rel_path)


def train_model() -> None:
    """
    Pipeline de treinamento do modelo:
      1. Carrega o DataFrame final com features e target.
      2. Prepara os conjuntos de treino e teste.
      3. Salva os dados de 'trusted_data' em formato Parquet.
      4. Valida o carregamento dos dados.
    """
    # 1. Carregar features finais com target
    final_feats_file = os.path.join("data", "features", "final_feats_with_target.parquet")
    logger.info("Carregando features finais com target de %s...", final_feats_file)
    final_feats = pd.read_parquet(final_feats_file)

    # 2. Preparar conjuntos de treino e teste
    logger.info("Preparando os conjuntos de treino e teste...")
    trusted_data = prepare_features(final_feats)

    # Criar diretório base para salvar os dados de treino
    train_base_path = os.path.join("data", "train")
    os.makedirs(train_base_path, exist_ok=True)

    # 3. Salvar os itens de trusted_data
    for key, data in trusted_data.items():
        # Converter para DataFrame, se necessário
        if key == "encoder_mapping" and isinstance(data, dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as error:
                logger.warning("Não foi possível converter '%s' para DataFrame: %s", key, error)
                continue

        file_name = f"{key}.parquet"
        file_path = os.path.join(train_base_path, file_name)
        logger.info("Salvando '%s'...", file_name)
        save_dataframe_as_parquet(data, file_path)

    logger.info("Pipeline de treinamento concluído!")

    # 4. Validar o carregamento dos dados
    logger.info("Validando o carregamento dos dados...")
    X_train, y_train = load_train_data()
    logger.info("Dados carregados: X_train shape: %s, y_train shape: %s", X_train.shape, y_train.shape)


if __name__ == "__main__":
    train_model()