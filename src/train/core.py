"""
Módulo core contendo funções comuns para treinamento de modelos.
Centraliza a lógica de treinamento para ser reutilizada em diferentes scripts.
"""
import os
import pandas as pd
import mlflow
from typing import Dict, Any, Optional

from config import logger, DATA_PATH, get_config
from features.schemas import get_model_signature, create_mock_input_example
from recomendation_model.base_model import BaseRecommender
from recomendation_model.mocked_model import MLflowWrapper


def log_model_to_mlflow(
    model: BaseRecommender,
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
    register: bool = True,
    set_as_champion: bool = True,
) -> str:
    """
    Registra um modelo no MLflow e opcionalmente no Model Registry.

    Args:
        model (BaseRecommender): Modelo treinado a ser registrado.
        model_name (str, opcional): Nome do modelo. Se None, usa a configuração.
        run_id (str, opcional): ID do experimento MLflow atual.
        register (bool): Se True, registra o modelo no Model Registry.
        set_as_champion (bool): Se True, define o modelo como "champion".

    Returns:
        str: URI do modelo registrado.
    """
    # Usa o nome do modelo da configuração se não for fornecido
    if model_name is None:
        model_name = get_config("MODEL_NAME", "news-recommender")

    # Cria um exemplo de entrada para o modelo
    input_example = create_mock_input_example()

    # Obtém a assinatura do modelo
    signature = get_model_signature()

    # Cria um wrapper MLflow para o modelo
    wrapper = MLflowWrapper(model)

    # Registra o modelo no MLflow
    mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=wrapper,
        signature=signature,
        input_example=input_example
    )

    # Se register=False, retorna URI sem registrar no Model Registry
    if not register or run_id is None:
        logger.info("Modelo salvo no MLflow sem registro no Model Registry")
        return ""

    # Registra o modelo no Model Registry
    model_uri = f"runs:/{run_id}/{model_name}"
    try:
        model_details = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        logger.info(f"Modelo registrado: {model_details.name}, versão {model_details.version}")

        # Define a versão como 'champion' se solicitado
        if set_as_champion:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(model_name, "champion", model_details.version)
            logger.info(
                f"""
                Alias 'champion' definido para
                 versão {model_details.version} do modelo {model_name}"""
            )
    except Exception as e:
        logger.warning(f"Não foi possível registrar o modelo: {e}")
        logger.info(f"URI do modelo: {model_uri}")

    # Log do run_id para referência
    logger.info(f"MLflow run_id: {run_id}")
    logger.info(f"URI do modelo registrado: {model_uri}")

    return model_uri


def load_model_from_mlflow(
    model_name: Optional[str] = None,
    model_alias: Optional[str] = None
) -> Any:
    """
    Carrega um modelo registrado no MLflow.

    Args:
        model_name (str, opcional): Nome do modelo. Se None, usa a configuração.
        model_alias (str, opcional): Alias do modelo. Se None, usa "champion".

    Returns:
        Any: Modelo carregado.
    """
    if model_name is None:
        model_name = get_config("MODEL_NAME", "news-recommender")

    if model_alias is None:
        model_alias = get_config("MODEL_ALIAS", "champion")

    model_uri = f"models:/{model_name}@{model_alias}"
    logger.info(f"Carregando modelo {model_uri} do MLflow")

    try:
        model = mlflow.pyfunc.load_model(model_uri)
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo {model_uri}: {e}")
        return None


def log_encoder_mapping(trusted_data: Dict[str, Any]) -> None:
    """
    Salva e registra o encoder_mapping como artefato do MLflow.

    Args:
        trusted_data (Dict[str, Any]): Dados processados contendo encoder_mapping.
    """
    train_base_path = os.path.join(DATA_PATH, "train")
    encoder_path = os.path.join(train_base_path, "encoder_mapping.json")
    pd.DataFrame(trusted_data["encoder_mapping"]).to_json(encoder_path)
    mlflow.log_artifact(encoder_path)


def log_basic_metrics(X_train: pd.DataFrame, metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Registra métricas básicas do treinamento no MLflow.

    Args:
        X_train (pd.DataFrame): Features de treinamento.
        metrics (Dict[str, float], opcional): Métricas adicionais para registrar.
    """
    # Registra métricas básicas
    mlflow.log_metric("training_samples", len(X_train))
    mlflow.log_metric("num_features", X_train.shape[1])

    # Registra métricas adicionais se fornecidas
    if metrics is not None:
        mlflow.log_metrics(metrics)


def get_run_name(model_name: Optional[str] = None) -> str:
    """
    Gera um nome de execução baseado no modelo e timestamp atual.

    Args:
        model_name (str, opcional): Nome do modelo para o run_name. Se None, usa a configuração.

    Returns:
        str: Nome de execução para o MLflow.
    """
    if model_name is None:
        model_name = get_config("MODEL_NAME", "news-recommender")

    # Gera um nome de execução baseado no timestamp
    run_name = f"{model_name}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
    return run_name
