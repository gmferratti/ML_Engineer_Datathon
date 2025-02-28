import mlflow
from typing import Dict
from src.config import configure_mlflow, get_config
from src.recomendation_model.mocked_model import MockedRecommender, MLflowWrapper
from src.recomendation_model.base_model import BaseRecommender
from src.features.schemas import get_model_signature, create_mock_input_example
from src.evaluation.utils import evaluate_model
from src.train.utils import load_train_data
from src.data.data_loader import get_evaluation_data


def train_model(model_params: Dict) -> BaseRecommender:
    """Treina o modelo de recomendacao.

    Args:
        model_params (Dict): Parametros do modelo.

    Returns:
        BaseRecommender: Modelo treinado.
    """
    X_train, y_train = load_train_data()
    evaluation_data = get_evaluation_data()

    input_example = create_mock_input_example()

    # Inicia um novo experimento
    with mlflow.start_run() as run:
        # Cria e treina o modelo
        model = MockedRecommender(**model_params)
        model.train(X_train, y_train)

        # Loga parâmetros
        mlflow.log_params(model_params)

        # Loga métricas
        metrics = evaluate_model(model, evaluation_data)
        mlflow.log_metrics(metrics)

        # Salva o modelo usando o wrapper e a signature
        wrapper = MLflowWrapper(model)
        mlflow.pyfunc.log_model(
            artifact_path=get_config('MODEL_NAME'),
            python_model=wrapper,
            signature=get_model_signature(),
            input_example=input_example
        )

        # Guarda o run_id
        run_id = run.info.run_id
        print(f"Modelo treinado com sucesso. Run ID: {run_id}")


if __name__ == "__main__":
    configure_mlflow()

    model_params = get_config('MODEL_PARAMS')
    trained_model = train_model(model_params=model_params)
