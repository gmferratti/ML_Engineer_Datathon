import os
import re
import random
import pandas as pd
from typing import Tuple, Dict, Any

from src.config import DATA_PATH, USE_S3, configure_mlflow, logger
from src.storage.io import Storage
from src.data.data_loader import load_data_for_prediction
from src.predict.pipeline import predict_for_userId
from src.train.core import load_model_from_mlflow


def explode_history(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte a coluna 'history' (contendo strings com pageIds colados)
    em múltiplas linhas, extraindo os valores entre aspas simples.

    Exemplo de entrada:
      "['be89a7da-d9fa-49d4-9fdc-388c27a15bc8'\n '01c59ff6-fb82-4258-918f-2910cb2d4c52']"

    Returns:
        pd.DataFrame: DataFrame com a coluna 'history' explodida e renomeada para 'pageId'.
    """

    def parse_history_str(s: str) -> list:
        s = s.strip().replace("[", "").replace("]", "")
        tokens = re.findall(r"'([^']+)'", s)
        return tokens

    df["history"] = df["history"].apply(
        lambda x: parse_history_str(x) if isinstance(x, str) else x
    )
    df_exploded = df.explode("history").reset_index(drop=True)
    df_exploded.rename(columns={"history": "pageId"}, inplace=True)
    return df_exploded


def evaluate_model(
    model: Any, n: int = 5, score_threshold: float = 15, sample_size: int = 5
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Avalia o modelo nos dados de validação.

    O fluxo realiza:
      1. Leitura do CSV de validação.
      2. Explosão da coluna 'history' para obter os pageIds consumidos (ground truth).
      3. Agrupamento por usuário para coletar o conjunto de pageIds consumidos.
      4. Carregamento dos DataFrames de predição (features de notícias e clientes).
      5. Para cada usuário (amostrado, se sample_size for definido), gera recomendações
         e verifica se há pelo menos um acerto (hit rate).

    Args:
        model: Modelo treinado para predição.
        n (int, optional): Número máximo de recomendações por usuário. Default: 5.
        score_threshold (float, optional): Score mínimo para considerar uma recomendação.
        Default: 15.
        sample_size (int, optional): Se definido, avalia apenas essa quantidade de usuários.

    Returns:
        Tuple[Dict[str, Any], Dict[str, Any]]:
            - Métricas agregadas (total de usuários avaliados, hits e hit_rate).
            - Detalhes por usuário (ground truth, recomendações e flag de hit).
    """
    storage = Storage(use_s3=USE_S3)
    validation_csv_path = os.path.join(
        DATA_PATH, "challenge-webmedia-e-globo-2023/val_data/validacao.csv"
    )
    val_df = storage.read_csv(validation_csv_path)
    logger.info("🔍 [Evaluation] Dados de validação carregados: %s", val_df.shape)

    # Explode a coluna 'history' para obter os pageIds consumidos
    val_exploded = explode_history(val_df)
    logger.info("🔍 [Evaluation] Dados de validação explodidos: %s", val_exploded.shape)

    # Agrupa por usuário para formar o ground truth
    ground_truth = val_exploded.groupby("userId")["pageId"].apply(set).to_dict()
    logger.info("🔍 [Evaluation] Número de usuários no ground truth: %d", len(ground_truth))

    # Amostra os usuários para acelerar a avaliação, se sample_size for especificado
    if sample_size and sample_size < len(ground_truth):
        all_users = list(ground_truth.keys())
        sampled_users = random.sample(all_users, sample_size)
        ground_truth = {user: ground_truth[user] for user in sampled_users}
        logger.info("🔍 [Evaluation] Amostrando ground truth para %d usuários.", sample_size)

    # Carrega os DataFrames de predição
    pred_data = load_data_for_prediction(storage, include_metadata=False)
    news_features_df = pred_data["news_features"]
    clients_features_df = pred_data["clients_features"]

    hits = 0
    total_users = 0
    details = {}

    # Para cada usuário, gera recomendações e verifica se há acerto
    for user, true_page_ids in ground_truth.items():
        recs, is_cold_start = predict_for_userId(
            user,
            clients_features_df,
            news_features_df,
            model,
            n=n,
            score_threshold=score_threshold,
        )
        rec_page_ids = {rec["pageId"] for rec in recs}
        hit = len(true_page_ids.intersection(rec_page_ids)) > 0
        if hit:
            hits += 1
        total_users += 1
        details[user] = {
            "ground_truth": true_page_ids,
            "recommendations": rec_page_ids,
            "hit": hit,
        }

    hit_rate = hits / total_users if total_users > 0 else 0.0
    metrics = {"total_users": total_users, "hits": hits, "hit_rate": hit_rate}
    return metrics, details


if __name__ == "__main__":
    configure_mlflow()
    model = load_model_from_mlflow()
    if model is None:
        logger.error("Modelo não carregado. Verifique os logs do MLflow.")
        raise SystemExit("Encerrando avaliação, modelo não disponível.")

    metrics, details = evaluate_model(model, n=5, score_threshold=15)
    logger.info("Métricas de avaliação: %s", metrics)
    print("Métricas de avaliação:", metrics)
