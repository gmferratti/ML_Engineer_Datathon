import os
import pandas as pd
from typing import Tuple, List

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

def predict_for_userId(userId: str,
                       clients_features_df: pd.DataFrame,
                       news_features_df: pd.DataFrame,
                       model,
                       n: int = 5,
                       score_threshold: float = 15) -> List[str]:
    """
    Realiza a predição e gera recomendações para o usuário.
    """
    final_input, non_viewed = build_model_input(userId, clients_features_df, news_features_df)
    if final_input.empty:
        logger.info("🙁 [Predict] Nenhum input construído para o usuário %s.", userId)
        return []
    
    # Realiza a predição
    scores = model.predict(final_input)
    logger.info("🔮 [Predict] Predição realizada para o usuário %s com %d scores.", userId, len(scores))
    
    # Gera as recomendações
    recommendations = get_predicted_news(scores, non_viewed, n=n, score_threshold=score_threshold)
    logger.info("🎯 [Predict] Recomendações geradas para o usuário %s.", userId)
    return recommendations

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
