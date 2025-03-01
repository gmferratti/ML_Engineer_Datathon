import os
import pandas as pd
import mlflow
from typing import Dict, Any, Optional
from src.config import logger, DATA_PATH, get_config
from src.features.schemas import get_model_signature, create_valid_input_example
from src.recommendation_model.base_model import BaseRecommender
from src.recommendation_model.mocked_model import MLflowWrapper


def log_model_to_mlflow(
    model: BaseRecommender,
    model_name: Optional[str] = None,
    run_id: Optional[str] = None,
    register: bool = True,
    set_as_champion: bool = True,
) -> str:
    """
    Registra o modelo no MLflow e opcionalmente no Model Registry.

    Args:
        model (BaseRecommender): Modelo treinado.
        model_name (str, optional): Nome do modelo. Default: configuração.
        run_id (str, optional): ID do experimento.
        register (bool): Se True, registra no Model Registry.
        set_as_champion (bool): Se True, define como "champion".

    Returns:
        str: URI do modelo registrado.
    """
    if model_name is None:
        model_name = get_config("MODEL_NAME", "news-recommender")
    input_ex = create_valid_input_example()
    signature = get_model_signature()
    wrapper = MLflowWrapper(model)
    mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=wrapper,
        signature=signature,
        input_example=input_ex,
    )
    if not register or run_id is None:
        logger.info("Modelo salvo sem registro no Model Registry")
        return ""
    model_uri = f"runs:/{run_id}/{model_name}"
    try:
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info("Modelo registrado: %s, versão %s", model_details.name, model_details.version)
        if set_as_champion:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(model_name, "champion", model_details.version)
            logger.info(
                "Alias 'champion' definido para versão %s do modelo %s",
                model_details.version,
                model_name,
            )
    except Exception as e:
        logger.warning("Não foi possível registrar o modelo: %s", e)
        logger.info("URI do modelo: %s", model_uri)
    logger.info("MLflow run_id: %s", run_id)
    logger.info("URI do modelo registrado: %s", model_uri)
    return model_uri


def load_model_from_mlflow(
    model_name: Optional[str] = None, model_alias: Optional[str] = None
) -> Any:
    """
    Carrega um modelo registrado no MLflow.

    Args:
        model_name (str, optional): Nome do modelo. Default: configuração.
        model_alias (str, optional): Alias do modelo. Default: "champion".

    Returns:
        Any: Modelo carregado ou None.
    """
    if model_name is None:
        model_name = get_config("MODEL_NAME", "news-recommender")
    if model_alias is None:
        model_alias = get_config("MODEL_ALIAS", "champion")
    model_uri = f"models:/{model_name}@{model_alias}"
    logger.info("Carregando modelo %s do MLflow", model_uri)
    try:
        return mlflow.pyfunc.load_model(model_uri)
    except Exception as e:
        logger.error("Erro ao carregar modelo %s: %s", model_uri, e)
        return None


def log_encoder_mapping(trusted_data: Dict[str, Any]) -> None:
    """
    Salva e registra o encoder_mapping como artefato do MLflow.

    Args:
        trusted_data (dict): Dados contendo encoder_mapping.
    """
    train_path = os.path.join(DATA_PATH, "train")
    encoder_path = os.path.join(train_path, "encoder_mapping.json")
    pd.DataFrame(trusted_data["encoder_mapping"]).to_json(encoder_path)
    mlflow.log_artifact(encoder_path)


def log_basic_metrics(X_train: pd.DataFrame, metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Registra métricas básicas de treinamento no MLflow.

    Args:
        X_train (pd.DataFrame): Features de treino.
        metrics (dict, optional): Métricas adicionais.
    """
    mlflow.log_metric("training_samples", len(X_train))
    mlflow.log_metric("num_features", X_train.shape[1])
    if metrics is not None:
        mlflow.log_metrics(metrics)


def get_run_name(model_name: Optional[str] = None) -> str:
    """
    Gera um nome de execução baseado no modelo e timestamp.

    Args:
        model_name (str, optional): Nome do modelo. Default: configuração.

    Returns:
        str: Nome da execução.
    """
    if model_name is None:
        model_name = get_config("MODEL_NAME", "news-recommender")
    return f"{model_name}-{pd.Timestamp.now().strftime('%Y%m%d-%H%M%S')}"
