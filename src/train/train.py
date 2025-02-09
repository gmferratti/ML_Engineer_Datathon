from typing import Dict

import mlflow

from src.data.data_loader import get_evaluation_data
from src.evaluation.utils import evaluate_model
from src.recomendation_model.base_model import BaseRecommender

from ..config import configure_mlflow, get_config
from ..recomendation_model.mocked_model import MockedRecommender
from .utils import load_train_data


def train_model(model_params: Dict) -> BaseRecommender:
    """Treina o modelo de recomendacao.

    Args:
        model_params (Dict): Parametros do modelo.

    Returns:
        BaseRecommender: Modelo treinado.
    """
    configure_mlflow()
    X_train, y_train = load_train_data()
    evaluation_data = get_evaluation_data()

    with mlflow.start_run():
        model = MockedRecommender(**model_params)
        model.train(X_train, y_train)

        mlflow.log_params(model_params)

        metrics = evaluate_model(model, evaluation_data)
        mlflow.log_metrics(metrics)

        mlflow.pyfunc.log_model(get_config("MODEL_NAME"), python_model=model)

        return model


if __name__ == "__main__":
    trained_model = train_model()
