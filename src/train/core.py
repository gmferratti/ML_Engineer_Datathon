import os
import tempfile
import pandas as pd
import mlflow
from typing import Dict, Any, Optional

from src.config import logger, get_config
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
    Registra o modelo no MLflow e, opcionalmente, no Model Registry.
    """
    input_ex = create_valid_input_example()
    signature = get_model_signature()
    wrapper = MLflowWrapper(model)

    logger.info("📦 [Core] Registrando modelo '%s' no MLflow...", model_name)
    # Definindo a tag com a versão para ser utilizada na extração dos metadados
    mlflow.set_tag("mlflow.runName", model_name)
    mlflow.pyfunc.log_model(
        artifact_path=model_name,
        python_model=wrapper,
        signature=signature,
        input_example=input_ex,
    )

    if not register or run_id is None:
        logger.info("ℹ️ [Core] Modelo salvo sem registro no Model Registry.")
        return ""

    model_uri = f"runs:/{run_id}/{model_name}"
    try:
        model_details = mlflow.register_model(model_uri=model_uri, name=model_name)
        logger.info(
            "✅ [Core] Modelo registrado: %s (versão: %s)",
            model_details.name,
            model_details.version,
        )
        if set_as_champion:
            client = mlflow.MlflowClient()
            client.set_registered_model_alias(model_name, "champion", model_details.version)
            logger.info(
                "🏆 [Core] Alias 'champion' definido para a versão %s do modelo %s.",
                model_details.version,
                model_name,
            )
    except Exception as e:
        logger.warning("🚨 [Core] Registro falhou: %s", e)
        logger.info("🔗 [Core] URI do modelo: %s", model_uri)

    logger.info("🔄 [Core] MLflow run_id: %s", run_id)
    logger.info("🔗 [Core] Modelo registrado: %s", model_uri)
    return model_uri


def load_model_from_mlflow(
    model_name: Optional[str] = None, model_alias: Optional[str] = None
) -> Any:
    """
    Carrega um modelo registrado no MLflow.
    """
    if model_name is None:
        model_name = get_config("MODEL_NAME")
        if model_name is None:
            logger.error("🚨 [Core] Nome do modelo não especificado.")
            raise ValueError("Nome do modelo não especificado.")
    if model_alias is None:
        model_alias = get_config("MODEL_ALIAS", "champion")

    model_uri = f"models:/{model_name}@{model_alias}"
    logger.info("🔄 [Core] Carregando modelo do MLflow: %s", model_uri)
    try:
        loaded_model = mlflow.pyfunc.load_model(model_uri)
        logger.info("✅ [Core] Modelo carregado com sucesso!")
        return loaded_model
    except Exception as e:
        logger.error("🚨 [Core] Erro ao carregar modelo %s: %s", model_uri, e)
        return None


def log_encoder_mapping(trusted_data: Dict[str, Any]) -> None:
    """
    Salva e registra o encoder_mapping como artefato no MLflow.

    Em vez de salvar direto em DATA_PATH (que pode ser S3),
    criamos um arquivo temporário local e chamamos mlflow.log_artifact.
    """
    df_map = pd.DataFrame(trusted_data["encoder_mapping"])
    if df_map.empty:
        logger.warning("🚨 [Core] O encoder_mapping está vazio. Nada para logar.")
        return

    # Cria um arquivo temporário local
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as tmp:
        local_json_path = tmp.name

    # Salva o DataFrame no arquivo temporário
    df_map.to_json(local_json_path)

    # Faz o log do artifact local com mlflow
    mlflow.log_artifact(local_json_path, artifact_path="encoder")

    # Remove o arquivo local temporário
    os.remove(local_json_path)

    logger.info("📝 [Core] Encoder mapping registrado.")


def log_metrics(X_train: pd.DataFrame, metrics: Optional[Dict[str, float]] = None) -> None:
    """
    Registra métricas básicas de treinamento no MLflow.
    """
    mlflow.log_metric("training_samples", len(X_train))
    mlflow.log_metric("num_features", X_train.shape[1])
    if metrics is not None:
        mlflow.log_metrics(metrics)
    logger.info("📊 [Core] Métricas de treinamento registradas.")
