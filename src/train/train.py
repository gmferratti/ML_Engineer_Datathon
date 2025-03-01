import mlflow
from typing import Dict
from src.config import configure_mlflow, get_config
from src.recommendation_model.lgbm_ranker import LightGBMRanker
from src.train.core import log_model_to_mlflow, log_basic_metrics, get_run_name
from src.evaluation.utils import evaluate_model
from src.train.utils import load_train_data
from src.data.data_loader import get_evaluation_data


def train_model(model_params: Dict = {}) -> LightGBMRanker:
    """
    Treina o modelo de recomendação.

    Args:
        model_params (dict, optional): Parâmetros do modelo.

    Returns:
        LightGBMRanker: Modelo treinado.
    """
    if model_params is None:
        model_params = get_config("MODEL_PARAMS", {})
    model_name = get_config("MODEL_NAME", "news-recommender")
    X_train, y_train = load_train_data()
    eval_data = get_evaluation_data()
    run_name = get_run_name(model_name)
    with mlflow.start_run(run_name=run_name) as run:
        model = LightGBMRanker(**model_params)
        model.train(X_train, y_train)
        mlflow.log_params(model_params)
        metrics = evaluate_model(model, eval_data)
        log_basic_metrics(X_train, metrics)
        log_model_to_mlflow(model, model_name, run.info.run_id)
        print(f"Modelo treinado. Run ID: {run.info.run_id}")

    return model


if __name__ == "__main__":
    configure_mlflow()
    params = get_config("MODEL_PARAMS", {})
    _ = train_model(model_params=params)
