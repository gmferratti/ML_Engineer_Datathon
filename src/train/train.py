"""
Script simplificado para treinamento de modelos de recomendação.
Ideal para experimentos rápidos e testes de diferentes modelos.
"""
import mlflow
from typing import Dict, Any, Optional

from config import configure_mlflow, get_config
from recomendation_model.mocked_model import MockedRecommender
from recomendation_model.base_model import BaseRecommender
from train.utils import load_train_data
from train.core import log_model_to_mlflow, log_basic_metrics, get_run_name
from data.data_loader import get_evaluation_data
from evaluation.utils import evaluate_model


def train_simple(
    model_params: Dict[str, Any] = None,
    model_class: Any = MockedRecommender,
    model_name: Optional[str] = None,
    evaluate: bool = True
) -> BaseRecommender:
    """
    Treina um modelo de recomendação usando MLflow para tracking.

    Args:
        model_params (Dict[str, Any], opcional): Parâmetros do modelo.
        model_class (Any): Classe do modelo a ser treinado.
        model_name (str, opcional): Nome do modelo. Se None, usa a configuração.
        evaluate (bool): Se True, avalia o modelo após o treinamento.

    Returns:
        BaseRecommender: Modelo treinado.
    """
    # Usa os parâmetros da configuração se não for fornecido
    if model_params is None:
        model_params = get_config("MODEL_PARAMS", {})

    # Usa o nome do modelo da configuração se não for fornecido
    if model_name is None:
        model_name = get_config("MODEL_NAME", "news-recommender")

    # Carrega os dados de treino
    X_train, y_train = load_train_data()

    # Carrega dados de avaliação se a avaliação for solicitada
    if evaluate:
        evaluation_data = get_evaluation_data()

    # Gera um nome para o experimento
    run_name = get_run_name(model_name)

    # Inicia um novo experimento com gerenciamento de contexto seguro
    with mlflow.start_run(run_name=run_name) as run:
        # Cria e treina o modelo com os parâmetros fornecidos
        model = model_class(**model_params)
        model.train(X_train, y_train)

        # Loga parâmetros no MLflow
        mlflow.log_params(model_params)

        # Avalia o modelo se solicitado
        if evaluate:
            metrics = evaluate_model(model, evaluation_data)
            mlflow.log_metrics(metrics)

        # Loga métricas básicas
        log_basic_metrics(X_train)

        # Salva o modelo usando o wrapper e a signature
        log_model_to_mlflow(model, model_name, run.info.run_id)

        # Imprime o run_id para referência
        run_id = run.info.run_id
        print(f"Modelo treinado com sucesso. Run ID: {run_id}")

    return model


if __name__ == "__main__":
    # Configura o MLflow
    configure_mlflow()

    # Obtém parâmetros do modelo da configuração
    model_params = get_config("MODEL_PARAMS", {})

    # Treina o modelo com a classe padrão (MockedRecommender)
    trained_model = train_simple(model_params=model_params)
