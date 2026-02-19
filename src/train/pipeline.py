import os
import pandas as pd
import mlflow
from typing import Dict, Any, Tuple

from src.train.utils import prepare_features, load_train_data
from src.train.core import (
    log_model_to_mlflow,
    log_encoder_mapping,
    log_metrics,
)
from src.evaluation.pipeline import evaluate_model
from src.config import logger, DATA_PATH, USE_S3, configure_mlflow, get_config
from src.recommendation_model.lgbm_ranker import LightGBMRanker
from src.storage.io import Storage


def load_features(storage: Storage) -> pd.DataFrame:
    """
    Carrega o DataFrame final de features e target.
    """
    file_path = os.path.join(DATA_PATH, "features", "final_feats_with_target.parquet")
    logger.info("🔍 [Train] Carregando features do arquivo: %s", file_path)
    df = storage.read_parquet(file_path)
    logger.info("📊 [Train] Features carregadas com shape: %s", df.shape)
    logger.info("✅ [Train] Features disponíveis: %s", df.columns.tolist())
    logger.info("🔍 [Train] Verificando dados ausentes...")
    logger.info("📊 [Train] Tipos das features de entrada: %s", df.dtypes.to_dict())
    return df


def prepare_and_save_train_data(storage: Storage, final_feats: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepara e salva os dados de treino.
    """
    logger.info("🔧 [Train] Preparando dados de treino...")
    trusted = prepare_features(final_feats)
    train_dir = os.path.join(DATA_PATH, "train")

    for key, data in trusted.items():
        if key == "encoder_mapping":
            continue
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)
        file_path = os.path.join(train_dir, f"{key}.parquet")
        logger.info("💾 [Train] Salvando '%s' em: %s | Shape: %s", key, file_path, data.shape)
        storage.write_parquet(data, file_path)

    logger.info("✅ [Train] Pipeline de features concluído!")
    return trusted


def validate_and_load_train_data(
    storage: Storage,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Valida e carrega os dados de treino a partir dos arquivos salvos.
    """
    logger.info("🔎 [Train] Validando dados de treino...")
    X_val, y_val = load_train_data(storage)
    logger.info(
        "📈 [Train] Dados provisórios: X_train: %s, y_train: %s", X_val.shape, y_val.shape
    )

    train_dir = os.path.join(DATA_PATH, "train")
    x_path = os.path.join(train_dir, "X_train.parquet")
    y_path = os.path.join(train_dir, "y_train.parquet")
    group_path = os.path.join(train_dir, "group_train.parquet")

    X_train = storage.read_parquet(x_path)
    y_train = storage.read_parquet(y_path)
    group_train = storage.read_parquet(group_path)

    logger.info(
        "✅ [Train] Dados carregados: X_train %s, y_train %s, group_train %s",
        X_train.shape,
        y_train.shape,
        group_train.shape,
    )
    return X_train, y_train, group_train


def train_and_log_model(
    X_train: pd.DataFrame,
    y_train: pd.DataFrame,
    group_train: pd.DataFrame,
    trusted_data: Dict[str, Any],
) -> None:
    """
    Treina o modelo LightGBMRanker e registra-o no MLflow.
    """
    params = get_config("MODEL_PARAMS", {})
    params.pop("threshold", None)  # Removendo 'threshold' para modelos de ranking.
    model_name = get_config("MODEL_NAME", "news-recommender-dev")

    logger.info("🚀 [Train] Iniciando treinamento do modelo '%s'...", model_name)
    with mlflow.start_run() as run:
        model = LightGBMRanker(params=params)
        model.train(X_train.values, y_train.values.ravel(), group_train["groupCount"].values)
        mlflow.log_params(params)
        log_encoder_mapping(trusted_data)

        metrics, _ = evaluate_model(model)
        log_metrics(X_train, metrics)
        log_model_to_mlflow(model, model_name, run.info.run_id)
        logger.info("🏁 [Train] Treinamento finalizado! MLflow run_id: %s", run.info.run_id)


def train_model_pipeline() -> None:
    """
    Pipeline principal de treinamento.
    """
    logger.info("=== 🚀 [Train] Iniciando Pipeline de Treinamento ===")
    storage = Storage(use_s3=USE_S3)

    final_feats = load_features(storage)
    trusted_data = prepare_and_save_train_data(storage, final_feats)
    X_train, y_train, group_train = validate_and_load_train_data(storage)

    configure_mlflow()
    train_and_log_model(X_train, y_train, group_train, trusted_data)

    logger.info("=== ✅ [Train] Pipeline de Treinamento Finalizado ===")


if __name__ == "__main__":
    train_model_pipeline()
