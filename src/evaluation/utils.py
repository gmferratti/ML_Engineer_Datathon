from typing import Dict
import pandas as pd
import numpy as np
from src.train.constants import CLIENT_FEATURES, NEWS_FEATURES
from src.evaluation.constants import EXPECTED_COLUMNS
from src.recommendation_model.lgbm_ranker import LightGBMRanker
from sklearn.metrics import ndcg_score


def evaluate_model(
    model: LightGBMRanker, evaluation_data: pd.DataFrame, k_ndcg: int = 10
) -> Dict:
    """
    Avalia o modelo de recomendação usando NDCG@k, considerando que evaluation_data 
    possui colunas de features e uma coluna 'TARGET' com a relevância.

    Args:
        model (LightGBMRanker): Modelo de recomendação treinado.
        evaluation_data (pd.DataFrame): DataFrame com colunas de features e a coluna 'TARGET'.
        k_ndcg (int, optional): Corte para a métrica NDCG. Default: 10.

    Returns:
        Dict: Dicionário com as métricas de avaliação, ex.: {"NDCG_10": valor}.
    """
    # Separar X (features) e y (target)
    X = evaluation_data.drop(columns=["TARGET"])
    y_true = evaluation_data["TARGET"].values

    # Obter arrays de features para clientes e notícias
    client_feats = X[CLIENT_FEATURES].values
    news_feats = X[NEWS_FEATURES].values

    # Combina as features horizontalmente para formar a matriz final
    final_features = np.concatenate([client_feats, news_feats], axis=1)

    # Cria um DataFrame com as colunas esperadas
    input_df = pd.DataFrame(final_features, columns=EXPECTED_COLUMNS)

    # Converte a coluna 'isWeekend' para booleano
    input_df["isWeekend"] = input_df["isWeekend"].astype(bool)

    # Para as demais colunas, converte explicitamente para float
    for col in EXPECTED_COLUMNS:
        if col != "isWeekend":
            input_df[col] = input_df[col].astype(float)

    # Obter predições do modelo utilizando o DataFrame formatado
    y_pred = model.predict(input_df)

    # Verifica se y_true e y_pred têm o mesmo tamanho
    assert len(y_true) == len(
        y_pred
    ), f"Dimensão incompatível: y_true tem {len(y_true)}, y_pred tem {len(y_pred)}"

    # Calcula o NDCG@k usando os arrays de verdadeiros e preditos
    ndcg_val = ndcg_score(np.array([y_true]), np.array([y_pred]), k=k_ndcg)
    metrics = {f"NDCG_{k_ndcg}": ndcg_val}
    return metrics
