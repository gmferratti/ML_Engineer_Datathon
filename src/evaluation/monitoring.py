import mlflow

from src.config import CONFIG
from src.data.data_loader import get_evaluation_data
from src.evaluation.utils import evaluate_model
from src.recomendation_model.mocked_model import MockedRecommender


def main():
    mlflow.set_tracking_uri(CONFIG.get("mlflow_tracking_uri"))
    model = MockedRecommender()  # TODO: Carregar modelo do MLflow
    evaluation_data = get_evaluation_data()
    metrics = evaluate_model(model, evaluation_data)
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
