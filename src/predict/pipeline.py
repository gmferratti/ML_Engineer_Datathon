import os
import pandas as pd
import datetime

from typing import Tuple, List, Dict, Any, Optional
from src.data.data_loader import load_data_for_prediction, get_client_features, get_predicted_news
from src.config import logger, configure_mlflow
from src.train.core import load_model_from_mlflow
from src.predict.constants import CLIENT_FEATURES_COLUMNS, NEWS_FEATURES_COLUMNS

def validate_features(df: pd.DataFrame, required_cols: List[str], source: str) -> None:
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error("🚨 [Predict] Colunas ausentes em %s: %s", source, missing)
        raise KeyError(f"Colunas ausentes em {source}: {missing}")
    logger.info("👍 [Predict] Todas as colunas necessárias foram encontradas em %s.", source)

def build_model_input(userId: str,
                      clients_features_df: pd.DataFrame,
                      news_features_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói o input final para o modelo baseado no usuário, utilizando os DataFrames
    separados de clientes e notícias.
    """
    # Obtém as features do cliente
    client_feat = get_client_features(userId, clients_features_df)
    if client_feat is None:
        logger.warning("⚠️ [Predict] Nenhuma feature encontrada para o usuário %s.", userId)
        return pd.DataFrame(), pd.DataFrame()
    
    client_df = pd.DataFrame([client_feat])
    logger.info("👤 [Predict] Features do cliente obtidas para o usuário %s.", userId)
    logger.info("📋 [Predict] Colunas do cliente: %s", client_df.columns.tolist())

    # Supondo que todas as notícias estão disponíveis (sem histórico de visualizações)
    non_viewed = news_features_df.copy()
    if non_viewed.empty:
        logger.warning("⚠️ [Predict] Nenhuma notícia disponível para o usuário %s.", userId)
        return pd.DataFrame(), non_viewed

    # Validação das colunas obrigatórias
    validate_features(client_df, CLIENT_FEATURES_COLUMNS, "Cliente")
    validate_features(non_viewed, NEWS_FEATURES_COLUMNS, "Notícias")

    # Extrai e prepara as features
    client_features = client_df[CLIENT_FEATURES_COLUMNS]
    news_features = non_viewed[NEWS_FEATURES_COLUMNS].reset_index(drop=True)
    logger.info("📑 [Predict] News features preparadas: %d registros.", len(news_features))

    # Repete as features do cliente para todas as notícias
    client_features_repeated = pd.concat([client_features] * len(news_features), ignore_index=True)
    logger.info("🔁 [Predict] Replicação das features do cliente para %d registros.", len(client_features_repeated))

    # Monta o DataFrame final para predição com a ordem esperada
    final_input = pd.DataFrame({
        'isWeekend': client_features_repeated['isWeekend'],
        'relLocalState': news_features['relLocalState'],
        'relLocalRegion': news_features['relLocalRegion'],
        'relThemeMain': news_features['relThemeMain'],
        'relThemeSub': news_features['relThemeSub'],
        'userTypeFreq': client_features_repeated['userTypeFreq'],
        'dayPeriodFreq': client_features_repeated['dayPeriodFreq'],
        'localStateFreq': news_features['localStateFreq'],
        'localRegionFreq': news_features['localRegionFreq'],
        'themeMainFreq': news_features['themeMainFreq'],
        'themeSubFreq': news_features['themeSubFreq']
    })

    logger.info("✅ [Predict] Input final preparado: %d registros, colunas: %s",
                len(final_input), final_input.columns.tolist())
    return final_input, non_viewed

def _handle_datetime_fields(row: pd.Series) -> Tuple[Optional[str], Optional[str]]:
    """
    Converte os campos 'issuedDate' e 'issuedTime' de uma linha para strings no formato ISO.

    Args:
        row (pd.Series): Linha de um DataFrame contendo as colunas 'issuedDate' e 'issuedTime'.

    Returns:
        Tuple[Optional[str], Optional[str]]: issuedDate e issuedTime formatados ou None.
    """
    issued_date = row.get("issuedDate")
    issued_time = row.get("issuedTime")
    issued_date_str: Optional[str] = None
    issued_time_str: Optional[str] = None

    if issued_date is not None:
        if isinstance(issued_date, (datetime.date, datetime.datetime)):
            issued_date_str = issued_date.isoformat()
        else:
            try:
                issued_date_str = pd.to_datetime(issued_date).date().isoformat()
            except Exception:
                issued_date_str = str(issued_date)
    if issued_time is not None:
        if isinstance(issued_time, (datetime.time, datetime.datetime)):
            issued_time_str = issued_time.isoformat()
        else:
            try:
                issued_time_str = pd.to_datetime(issued_time).time().isoformat()
            except Exception:
                issued_time_str = str(issued_time)
    return issued_date_str, issued_time_str


def _generate_cold_start_recommendations(news_features_df: pd.DataFrame,
                                        n: int) -> List[Dict[str, Any]]:
    """
    Gera recomendações para cold start, selecionando as notícias mais recentes
    com score fixo "desconhecido".

    Args:
        news_features_df (pd.DataFrame): DataFrame com as features das notícias.
        n (int): Número máximo de recomendações.

    Returns:
        List[Dict[str, Any]]: Lista de recomendações com score fixo e metadados.
    """
    # Cria coluna combinada 'issuedDatetime' se possível para ordenação
    if "issuedDate" in news_features_df.columns and "issuedTime" in news_features_df.columns:
        temp_df = news_features_df.copy()
        temp_df["issuedDatetime"] = pd.to_datetime(
            temp_df["issuedDate"].astype(str) + " " + temp_df["issuedTime"].astype(str),
            errors="coerce"
        )
        sorted_df = temp_df.sort_values("issuedDatetime", ascending=False).head(n)
    else:
        sorted_df = news_features_df.head(n)

    recommendations = []
    for _, row in sorted_df.iterrows():
        issued_date_str, issued_time_str = _handle_datetime_fields(row)
        recommendations.append({
            "pageId": row["pageId"],
            "score": "desconhecido",
            "title": row.get("title"),
            "url": row.get("url"),
            "issuedDate": issued_date_str,
            "issuedTime": issued_time_str
        })
    return recommendations


def _generate_normal_recommendations(scores: List[float],
                                    non_viewed: pd.DataFrame,
                                    news_features_df: pd.DataFrame,
                                    score_threshold: float,
                                    n: int) -> List[Dict[str, Any]]:
    """
    Gera recomendações para usuários não cold start, usando os scores gerados pelo modelo.

    Args:
        scores (List[float]): Lista de scores previstos.
        non_viewed (pd.DataFrame): DataFrame de notícias não vistas.
        news_features_df (pd.DataFrame): DataFrame original com metadados.
        score_threshold (float): Score mínimo para considerar uma recomendação.
        n (int): Número máximo de recomendações.

    Returns:
        List[Dict[str, Any]]: Lista de recomendações com metadados.
    """
    rec_entries = get_predicted_news(scores, non_viewed, n=n, score_threshold=score_threshold)
    recommendations = []
    for entry in rec_entries:
        news_id = entry.get("pageId")
        score = entry.get("score", 0)
        row = news_features_df[news_features_df["pageId"] == news_id]
        if not row.empty:
            title = row.iloc[0].get("title")
            url = row.iloc[0].get("url")
            issued_date_str, issued_time_str = _handle_datetime_fields(row.iloc[0])
        else:
            title, url, issued_date_str, issued_time_str = None, None, None, None

        recommendations.append({
            "pageId": news_id,
            "score": score,
            "title": title,
            "url": url,
            "issuedDate": issued_date_str,
            "issuedTime": issued_time_str
        })
    return recommendations


def predict_for_userId(userId: str,
                       clients_features_df: pd.DataFrame,
                       news_features_df: pd.DataFrame,
                       model,
                       n: int = 5,
                       score_threshold: float = 15) -> Tuple[List[Dict[str, Any]], bool]:
    """
    Realiza a predição e gera recomendações para o usuário.

    Se as features do usuário não forem encontradas (mas o userId for um hash válido),
    assume-se que ele é cold start e retorna as notícias mais recentes com score "desconhecido"
    e metadados ordenados do mais recente para o mais distante.

    Args:
        userId (str): ID do usuário.
        clients_features_df (pd.DataFrame): DataFrame com features dos clientes.
        news_features_df (pd.DataFrame): DataFrame com features das notícias.
        model: Modelo para predição.
        n (int): Número máximo de recomendações.
        score_threshold (float): Score mínimo para considerar uma recomendação.

    Returns:
        Tuple[List[Dict[str, Any]], bool]: Uma tupla contendo:
            - Lista de recomendações (cada recomendação contém 'pageId', 'score', 'title',
              'url', 'issuedDate' e 'issuedTime').
            - Flag que indica se o usuário é cold start.
    """
    # Tenta obter as features do cliente
    client_feat = get_client_features(userId, clients_features_df)

    # Se não encontrar e o userId tiver tamanho indicativo de hash, assume cold start
    if client_feat is None and len(userId) >= 64:
        logger.info("❄️ [Predict] Usuário %s não encontrado (hash válido). Assumindo cold start.", userId)
        recommendations = _generate_cold_start_recommendations(news_features_df, n)
        return recommendations, True

    # Fluxo normal
    final_input, non_viewed = build_model_input(userId, clients_features_df, news_features_df)
    if final_input.empty:
        logger.info("🙁 [Predict] Nenhum input construído para o usuário %s.", userId)
        return [], False

    scores = model.predict(final_input)
    logger.info("🔮 [Predict] Predição realizada para o usuário %s com %d scores.", userId, len(scores))
    recommendations = _generate_normal_recommendations(scores, non_viewed, news_features_df, score_threshold, n)
    return recommendations, False

def main():
    logger.info("=== 🚀 [Predict] Iniciando Pipeline de Predição ===")
    # Carrega os dados via data_loader (que agora retorna um dicionário com os DataFrames)
    data = load_data_for_prediction()
    news_features_df = data["news_features"]
    clients_features_df = data["clients_features"]
    
    configure_mlflow()
    model = load_model_from_mlflow()
    
    userId = "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297"
    logger.info("=== 🚀 [Predict] Processando predição para o usuário: %s ===", userId)
    
    recommendations = predict_for_userId(userId, clients_features_df, news_features_df, model)
    
    if recommendations:
        logger.info("👍 [Predict] Recomendações para o usuário %s: %s", userId, recommendations)
        print("🔔 Recomendações:")
        for rec in recommendations:
            print(" -", rec)
    else:
        logger.info("😕 [Predict] Nenhuma recomendação gerada para o usuário %s.", userId)
    
    logger.info("=== ✅ [Predict] Pipeline de Predição Finalizado ===")

if __name__ == "__main__":
    main()
