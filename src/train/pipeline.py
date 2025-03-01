"""
Pipeline principal de treinamento do modelo.
"""
import os
import pandas as pd
import mlflow
from typing import Dict, Any, Tuple

from train.utils import prepare_features, load_train_data
from train.core import (
    log_model_to_mlflow, log_encoder_mapping, log_basic_metrics, get_run_name
)
from config import logger, DATA_PATH, USE_S3, configure_mlflow, get_config
from recomendation_model.base_model import LightGBMRanker
from storage.io import Storage


def load_features(storage: Storage) -> pd.DataFrame:
    """
    Carrega os dados de features e target.

    Args:
        storage (Storage): Instância de Storage para operações de I/O.

    Returns:
        pd.DataFrame: DataFrame com features e target.
    """
    final_feats_file = os.path.join(DATA_PATH, "features", "final_feats_with_target.parquet")
    logger.info("Carregando features finais com target de %s...", final_feats_file)
    final_feats = storage.read_parquet(final_feats_file)
    logger.info("Shape do dataframe final_feats (antes do split): %s", final_feats.shape)
    return final_feats


def prepare_and_save_train_data(storage: Storage, final_feats: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepara os dados para treino e teste e salva em disco.

    Args:
        storage (Storage): Instância de Storage para operações de I/O.
        final_feats (pd.DataFrame): DataFrame com features e target.

    Returns:
        Dict[str, Any]: Dicionário com os conjuntos de dados preparados.
    """
    logger.info("Preparando os conjuntos de treino e teste (split e remoção do cold_start)...")
    trusted_data = prepare_features(final_feats)

    # Log dos shapes após o split e remoção do cold_start para cada item em trusted_data
    if isinstance(trusted_data, dict):
        for key, data in trusted_data.items():
            try:
                # Converter para DataFrame se necessário para garantir a obtenção do shape
                data_df = data if isinstance(data, pd.DataFrame) else pd.DataFrame(data)
                logger.info(
                    "Shape do dataframe '%s' (após split): %s", key, data_df.shape)
            except Exception as error:
                logger.warning("Não foi possível determinar o shape de '%s': %s", key, error)

    # Criar diretório base para salvar os dados de treino
    train_base_path = os.path.join(DATA_PATH, "train")

    # Salvar cada item de trusted_data em formato Parquet (exceto encoder_mapping)
    for key, data in trusted_data.items():
        # Pular encoder_mapping - será registrado como um artefato do MLflow
        if key == "encoder_mapping":
            continue

        # Para os demais itens, salvar como Parquet
        if not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as error:
                logger.warning("Não foi possível converter '%s' para DataFrame: %s", key, error)
                continue

        file_name = f"{key}.parquet"
        file_path = os.path.join(train_base_path, file_name)
        logger.info("Salvando '%s' em %s com shape: %s", key, file_path, data.shape)
        storage.write_parquet(data, file_path)

    logger.info("Pipeline de preparação de features concluído!")
    return trusted_data


def validate_and_load_train_data(
    storage: Storage
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Valida o carregamento dos dados de treino e carrega os dados necessários.

    Args:
        storage (Storage): Instância de Storage para operações de I/O.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: X_train, y_train e group_train.
    """
    # Validar o carregamento dos dados de treino
    logger.info("Validando o carregamento dos dados...")
    X_train_val, y_train_val = load_train_data(storage)
    logger.info("Dados carregados: X_train shape: %s, y_train shape: %s",
                X_train_val.shape, y_train_val.shape)

    # Carregar os dados para treino do modelo
    logger.info("Carregando dados para treino do modelo...")
    train_base_path = os.path.join(DATA_PATH, "train")
    x_train_path = os.path.join(train_base_path, "X_train.parquet")
    y_train_path = os.path.join(train_base_path, "y_train.parquet")
    group_train_path = os.path.join(train_base_path, "group_train.parquet")

    X_train = storage.read_parquet(x_train_path)
    y_train = storage.read_parquet(y_train_path)
    group_train = storage.read_parquet(group_train_path)
    logger.info(
        "Dados carregados: X_train: %s, y_train: %s, group_train: %s",
        X_train.shape, y_train.shape, group_train.shape
    )

    return X_train, y_train, group_train


def train_and_log_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    group_train: pd.DataFrame,
    trusted_data: Dict[str, Any]
) -> None:
    """
    Treina o modelo e registra no MLflow.

    Args:
        X_train (pd.DataFrame): Features de treino.
        y_train (pd.DataFrame): Target de treino.
        group_train (pd.DataFrame): Grupos de treino.
        trusted_data (Dict[str, Any]): Dados processados com encoder_mapping.
    """
    # Obtém parâmetros do modelo da configuração
    model_params = get_config("MODEL_PARAMS", {})
    model_name = get_config("MODEL_NAME", "news-recommender")

    # Gera um nome para o experimento
    run_name = get_run_name(model_name)

    # Inicia o experimento MLflow com gerenciamento de contexto seguro
    with mlflow.start_run(run_name=run_name) as run:
        logger.info("Treinando o modelo LightGBMRanker com MLflow tracking...")

        # Treina o modelo
        model = LightGBMRanker(params=model_params)
        model.train(
            X_train.values,
            y_train.values.ravel(),
            group_train["groupCount"].values
        )

        # Registra parâmetros no MLflow
        mlflow.log_params(model_params)

        # Registra encoder_mapping como artefato JSON
        log_encoder_mapping(trusted_data)

        # Registra métricas básicas
        log_basic_metrics(X_train)

        # Registra o modelo no MLflow
        log_model_to_mlflow(model, model_name, run.info.run_id)

    logger.info("Treinamento do modelo concluído com sucesso!")


def train_model() -> None:
    """
    Pipeline principal de treinamento do modelo:
      1. Configura MLflow e storage
      2. Carrega o DataFrame final com features e target
      3. Prepara os conjuntos de treino e teste e salva em disco
      4. Valida o carregamento dos dados de treino
      5. Treina o modelo LightGBMRanker e salva no MLflow
    """
    # Configura o MLflow
    configure_mlflow()

    # Inicializa o serviço de storage
    storage = Storage(use_s3=USE_S3)

    # 1. Carregar features finais com target
    final_feats = load_features(storage)

    # 2. Preparar conjuntos de treino e teste (split e remoção do cold_start)
    trusted_data = prepare_and_save_train_data(storage, final_feats)

    # 3. Validar o carregamento e carregar dados para treino
    X_train, y_train, group_train = validate_and_load_train_data(storage)

    # 4. Treinar e registrar o modelo
    train_and_log_model(X_train, y_train, group_train, trusted_data)


if __name__ == "__main__":
    train_model()
