import pandas as pd
import mlflow
from src.config import CONFIG
from src.recomendation_model.mocked_model import MockedModel
from src.recomendation_model.base_model import BaseModel
from src.data.data_loader import get_evaluation_data


def evaluate_model(
    model: BaseModel,
    evaluation_data: pd.DataFrame,
    **kwargs
):
    metrics = {"example_metric": 0.0}
    return metrics


def main():
    mlflow.set_tracking_uri(CONFIG.get("mlflow_tracking_uri"))
    model = MockedModel()
    evaluation_data = get_evaluation_data()
    metrics = evaluate_model(model, evaluation_data)
    print("Evaluation metrics:", metrics)


if __name__ == "__main__":
    main()
