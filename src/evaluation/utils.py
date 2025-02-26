from typing import Dict

import pandas as pd

from recomendation_model.base_model import BaseRecommender


def evaluate_model(model: BaseRecommender, evaluation_data: pd.DataFrame, **kwargs) -> Dict:
    """Avalia o modelo de recomendacao.

    Args:
        model (BaseRecommender): Modelo de recomendacao treinado.
        evaluation_data (pd.DataFrame): Dados de avaliacao.

    Returns:
        Dict: Dicionario com as metricas de avaliacao.
    """
    # TODO: Implementar metricas de avaliacao
    metrics = {"example_metric": 0.0}
    return metrics
