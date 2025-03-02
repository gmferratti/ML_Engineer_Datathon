import pandas as pd
import datetime
import time
from typing import Tuple, List, Dict, Any, Optional
from functools import lru_cache

from src.data.data_loader import load_data_for_prediction, get_client_features, get_predicted_news
from src.config import logger, configure_mlflow
from src.train.core import load_model_from_mlflow
from src.predict.constants import CLIENT_FEATURES_COLUMNS, NEWS_FEATURES_COLUMNS


def validate_features(df: pd.DataFrame, required_cols: List[str], source: str) -> None:
    """Valida as colunas necessárias no DataFrame"""
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error("🚨 [Predict] Colunas ausentes em %s: %s", source, missing)
        raise KeyError(f"Colunas ausentes em {source}: {missing}")
    logger.info("👍 [Predict] Todas as colunas necessárias foram encontradas em %s.", source)


def build_model_input(
    userId: str, clients_features_df: pd.DataFrame, news_features_df: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói o input final para o modelo baseado no usuário.
    Versão otimizada para melhorar performance.
    """
    start_time = time.time()

    # Obtém as features do cliente
    client_feat = get_client_features(userId, clients_features_df)
    if client_feat is None:
        logger.warning("⚠️ [Predict] Nenhuma feature encontrada para o usuário %s.", userId)
        return pd.DataFrame(), pd.DataFrame()

    client_get_time = time.time() - start_time
    logger.debug(f"Tempo para obter features do cliente: {client_get_time:.3f}s")

    # Cria o DataFrame do cliente uma única vez com apenas as colunas necessárias
    client_df = pd.DataFrame([{col: client_feat[col] for col in CLIENT_FEATURES_COLUMNS}])

    # Supondo que todas as notícias estão disponíveis para recomendação
    non_viewed = news_features_df.copy()
    if non_viewed.empty:
        logger.warning("⚠️ [Predict] Nenhuma notícia disponível para o usuário %s.", userId)
        return pd.DataFrame(), non_viewed

    prep_time = time.time()

    # Extrai apenas as colunas necessárias das notícias (evita cópia desnecessária de dados)
    news_features = non_viewed[NEWS_FEATURES_COLUMNS].reset_index(drop=True)

    # Otimização importante: em vez de usar pd.concat para repetir as features do cliente,
    # cria um DataFrame apenas com os valores repetidos usando NumPy
    num_news = len(news_features)

    # Criamos uma lista com os valores de cada coluna repetidos de acordo com o número de notícias
    client_cols = {}
    for col in CLIENT_FEATURES_COLUMNS:
        # Repetimos o valor para cada notícia
        client_cols[col] = [client_df[col].iloc[0]] * num_news

    # Agora criamos um DataFrame diretamente com essas colunas sem usar concat
    client_features_repeated = pd.DataFrame(client_cols)

    logger.debug(f"Tempo para preparar dataframes: {time.time() - prep_time:.3f}s")

    merge_time = time.time()

    # Constrói o DataFrame final combinando client_features_repeated com news_features
    # Em vez de criar um novo DataFrame, atualizamos o existente para evitar cópia
    final_input = pd.DataFrame()

    # Para cada coluna de cliente
    for col in CLIENT_FEATURES_COLUMNS:
        final_input[col] = client_features_repeated[col]

    # Para cada coluna de notícia
    for col in NEWS_FEATURES_COLUMNS:
        final_input[col] = news_features[col]

    logger.debug(f"Tempo para merge: {time.time() - merge_time:.3f}s")

    total_time = time.time() - start_time
    logger.info(
        "✅ [Predict] Input final preparado em %.3fs: %d registros", total_time, len(final_input)
    )

    return final_input, non_viewed


# Otimizamos com cache LRU para evitar processamentos repetidos de campos de data/hora
@lru_cache(maxsize=1024)
def _handle_datetime_fields_cached(date_val, time_val) -> Tuple[Optional[str], Optional[str]]:
    """
    Versão em cache da função _handle_datetime_fields que trabalha com valores simples
    ao invés de rows inteiras para melhorar a performance de cache.
    """
    issued_date_str: Optional[str] = None
    issued_time_str: Optional[str] = None

    if date_val is not None:
        if isinstance(date_val, (datetime.date, datetime.datetime)):
            issued_date_str = date_val.isoformat()
        else:
            try:
                issued_date_str = pd.to_datetime(date_val).date().isoformat()
            except Exception:
                issued_date_str = str(date_val)

    if time_val is not None:
        if isinstance(time_val, (datetime.time, datetime.datetime)):
            issued_time_str = time_val.isoformat()
        else:
            try:
                issued_time_str = pd.to_datetime(time_val).time().isoformat()
            except Exception:
                issued_time_str = str(time_val)

    return issued_date_str, issued_time_str


def _handle_datetime_fields(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    """
    Wrapper que usa a versão em cache para processar os campos de data/hora.
    """
    date_val = row.get("issuedDate")
    time_val = row.get("issuedTime")

    # Converte para hashable types se necessário para funcionamento do cache
    if isinstance(date_val, pd.Timestamp):
        date_val = date_val.to_pydatetime()
    if isinstance(time_val, pd.Timestamp):
        time_val = time_val.to_pydatetime()

    return _handle_datetime_fields_cached(date_val, time_val)


def _generate_cold_start_recommendations(
    news_features_df: pd.DataFrame, n: int
) -> List[Dict[str, Any]]:
    """
    Versão otimizada para gerar recomendações para cold start.
    """
    start_time = time.time()

    # Cria coluna combinada 'issuedDatetime' se possível para ordenação
    if "issuedDate" in news_features_df.columns and "issuedTime" in news_features_df.columns:
        # Otimização: não criar cópia completa do DataFrame, apenas as colunas necessárias
        cols_needed = ["pageId", "issuedDate", "issuedTime"]
        if "title" in news_features_df.columns:
            cols_needed.append("title")
        if "url" in news_features_df.columns:
            cols_needed.append("url")

        temp_df = news_features_df[cols_needed].copy()

        # Usa pd.to_datetime mais direto para melhorar performance
        temp_df["issuedDatetime"] = pd.to_datetime(
            temp_df["issuedDate"].astype(str) + " " + temp_df["issuedTime"].astype(str),
            errors="coerce",
        )
        sorted_df = temp_df.sort_values("issuedDatetime", ascending=False).head(n)
    else:
        sorted_df = news_features_df.head(n)

    recommendations = []
    for _, row in sorted_df.iterrows():
        issued_date_str, issued_time_str = _handle_datetime_fields(row)
        recommendations.append(
            {
                "pageId": row["pageId"],
                "score": "desconhecido",
                "title": row.get("title"),
                "url": row.get("url"),
                "issuedDate": issued_date_str,
                "issuedTime": issued_time_str,
            }
        )

    logger.debug(f"Cold start recommendations generated in: {time.time() - start_time:.3f}s")
    return recommendations


def _generate_normal_recommendations(
    scores: List[float],
    non_viewed: pd.DataFrame,
    news_features_df: pd.DataFrame,
    score_threshold: float,
    n: int,
) -> List[Dict[str, Any]]:
    """
    Versão otimizada para gerar recomendações para usuários não cold start.
    """
    start_time = time.time()

    rec_entries = get_predicted_news(scores, non_viewed, n=n, score_threshold=score_threshold)
    recommendations = []

    # Otimização: criar um dicionário para lookup rápido de metadados
    news_lookup = {}
    if len(rec_entries) > 0:
        # Pega apenas os pageIds das recomendações
        recommended_ids = [entry["pageId"] for entry in rec_entries]

        # Filtrar o DataFrame original para obter apenas as linhas necessárias
        filtered_news = news_features_df[news_features_df["pageId"].isin(recommended_ids)]

        # Criar lookup table para acesso rápido
        for _, row in filtered_news.iterrows():
            news_id = row["pageId"]
            issued_date_str, issued_time_str = _handle_datetime_fields(row)
            news_lookup[news_id] = {
                "title": row.get("title"),
                "url": row.get("url"),
                "issuedDate": issued_date_str,
                "issuedTime": issued_time_str,
            }

    # Constrói as recomendações usando o lookup
    for entry in rec_entries:
        news_id = entry["pageId"]
        score = entry.get("score", 0)

        metadata = news_lookup.get(news_id, {})

        recommendations.append(
            {
                "pageId": news_id,
                "score": score,
                "title": metadata.get("title"),
                "url": metadata.get("url"),
                "issuedDate": metadata.get("issuedDate"),
                "issuedTime": metadata.get("issuedTime"),
            }
        )

    logger.debug(f"Normal recommendations generated in: {time.time() - start_time:.3f}s")
    return recommendations


def predict_for_userId(
    userId: str,
    clients_features_df: pd.DataFrame,
    news_features_df: pd.DataFrame,
    model,
    n: int = 5,
    score_threshold: float = 15,
) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Versão otimizada para realizar a predição e gerar recomendações para o usuário.
    """
    start_total = time.time()

    # Tenta obter as features do cliente
    client_feat = get_client_features(userId, clients_features_df)

    # Se não encontrar e o userId tiver tamanho indicativo de hash, assume cold start
    if client_feat is None and len(userId) >= 64:
        logger.info(
            "❄️ [Predict] Usuário %s não encontrado (hash válido). Assumindo cold start.", userId
        )
        recommendations = _generate_cold_start_recommendations(news_features_df, n)
        total_time = time.time() - start_total
        logger.info(f"Predição cold start concluída em {total_time:.3f}s")
        return recommendations, True

    # Fluxo normal de predição
    start_input = time.time()
    final_input, non_viewed = build_model_input(userId, clients_features_df, news_features_df)
    input_time = time.time() - start_input

    if final_input.empty:
        logger.info("🙁 [Predict] Nenhum input construído para o usuário %s.", userId)
        return [], False

    # Medição do tempo de predição do modelo
    start_predict = time.time()
    scores = model.predict(final_input)
    predict_time = time.time() - start_predict

    logger.info(
        "🔮 [Predict] Predição realizada para o usuário %s com %d scores em %.3fs.",
        userId,
        len(scores),
        predict_time,
    )

    # Geração de recomendações
    start_rec = time.time()
    recommendations = _generate_normal_recommendations(
        scores, non_viewed, news_features_df, score_threshold, n
    )
    rec_time = time.time() - start_rec

    total_time = time.time() - start_total
    logger.info(
        "⏱️ [Predict] Tempos: input=%.3fs, predição=%.3fs, recomendações=%.3fs, total=%.3fs",
        input_time,
        predict_time,
        rec_time,
        total_time,
    )

    return recommendations, False


def main():
    logger.info("=== 🚀 [Predict] Iniciando Pipeline de Predição ===")
    # Carrega os dados via data_loader
    data = load_data_for_prediction()
    news_features_df = data["news_features"]
    clients_features_df = data["clients_features"]

    configure_mlflow()
    model = load_model_from_mlflow()

    userId = "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297"
    logger.info("=== 🚀 [Predict] Processando predição para o usuário: %s ===", userId)

    start_time = time.time()
    recommendations, is_cold_start = predict_for_userId(
        userId, clients_features_df, news_features_df, model
    )
    elapsed = time.time() - start_time

    logger.info("🥶 [Predict] Cold start: %s", is_cold_start)
    logger.info("⏱️ [Predict] Tempo total de predição: %.3f segundos", elapsed)

    if recommendations:
        logger.info("🔔 Recomendações:")
        for rec in recommendations:
            pageId = rec["pageId"]
            score = rec["score"]
            logger.info(f"{pageId} : {score}")
    else:
        logger.info("😕 [Predict] Nenhuma recomendação gerada para o usuário %s.", userId)

    logger.info("=== ✅ [Predict] Pipeline de Predição Finalizado ===")


if __name__ == "__main__":
    main()
