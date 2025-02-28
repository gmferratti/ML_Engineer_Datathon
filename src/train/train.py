import mlflow
from typing import Dict

from config import configure_mlflow, get_config
from recomendation_model.mocked_model import MockedRecommender, MLflowWrapper
from recomendation_model.base_model import LightGBMRanker
from features.schemas import get_model_signature, create_mock_input_example, create_valid_input_example
from evaluation.utils import evaluate_model
from utils import load_train_data
from data.data_loader import get_evaluation_data

def train_model(model_params : Dict = {}) -> LightGBMRanker:
    """Treina o modelo de recomendacao.

    Args:
        model_params (Dict): Parametros do modelo.

    Returns:
        BaseRecommender: Modelo treinado.
    """
    X_train, y_train = load_train_data()
    evaluation_data = get_evaluation_data()
    signature = get_model_signature()
    input_example = create_valid_input_example()

    # Inicia um novo experimento
    with mlflow.start_run() as run:
        # Cria e treina o modelo
        model = LightGBMRanker(**model_params)
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
            signature=signature,
            input_example=input_example,
            pip_requirements='requirements.txt' # só para evitar warnings. Atualize com pip freeze
        )

        # Guarda o run_id
        run_id = run.info.run_id
        print(f"Modelo treinado com sucesso. Run ID: {run_id}")


if __name__ == "__main__":
    configure_mlflow()

    model_params = get_config('MODEL_PARAMS')
    trained_model = train_model(model_params=model_params)
