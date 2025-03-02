import os
import pandas as pd
from typing import Tuple, List

from src.data.data_loader import get_client_features, get_predicted_news
from src.config import DATA_PATH, logger, USE_S3, configure_mlflow
from src.train.core import load_model_from_mlflow
from storage.io import Storage
from src.predict.constants import EXPECTED_COLUMNS, CLIENT_FEATURES_COLUMNS, NEWS_FEATURES_COLUMNS


def prepare_for_prediction() -> str:
    """
    Lê o DataFrame de features completo e o salva na pasta de predição.
    
    Returns:
        str: Caminho completo do arquivo salvo.
    """
    storage = Storage(use_s3=USE_S3)
    full_path = os.path.join(DATA_PATH, "train", "X_train_full.parquet")
    logger.info("🔍 [Predict] Carregando features de: %s", full_path)
    df = storage.read_parquet(full_path)
    
    pred_dir = os.path.join(DATA_PATH, "predict")
    os.makedirs(pred_dir, exist_ok=True)
    
    save_path = os.path.join(pred_dir, "X_train_full.parquet")
    storage.write_parquet(df, save_path)
    logger.info("✅ [Predict] Dados salvos em: %s", save_path)
    
    return save_path


def load_prediction_data(pred_path: str) -> pd.DataFrame:
    """
    Carrega o DataFrame de predição.
    
    Args:
        pred_path (str): Caminho do arquivo.
    
    Returns:
        pd.DataFrame: Dados carregados.
    """
    storage = Storage(use_s3=USE_S3)
    logger.info("🔄 [Predict] Carregando dados de: %s", pred_path)
    df = storage.read_parquet(pred_path)
    logger.info("📊 [Predict] Dados carregados: %d registros", len(df))
    return df


def validate_features(df: pd.DataFrame, required_cols: List[str], source: str) -> None:
    """
    Verifica se as colunas necessárias estão presentes no DataFrame.
    
    Args:
        df (pd.DataFrame): DataFrame a validar.
        required_cols (List[str]): Colunas obrigatórias.
        source (str): Origem (ex.: 'Cliente' ou 'Notícias').
    
    Raises:
        KeyError: Se alguma coluna estiver ausente.
    """
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        logger.error("🚨 [Predict] Colunas ausentes em %s: %s", source, missing)
        raise KeyError(f"Colunas ausentes em {source}: {missing}")
    logger.info("👍 [Predict] Todas as colunas necessárias foram encontradas em %s.", source)


def build_model_input(user_id: str, full_df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Constrói o input final para o modelo baseado no usuário.
    
    Args:
        user_id (str): ID do usuário.
        full_df (pd.DataFrame): DataFrame completo de features.
    
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]:
            - final_input: DataFrame para predição.
            - non_viewed: Dados originais de notícias não visualizadas.
    """
    # Seleciona notícias não visualizadas pelo usuário
    seen = full_df.loc[full_df["userId"] == user_id, "pageId"].unique()
    non_viewed = full_df[~full_df["pageId"].isin(seen)].copy()
    if non_viewed.empty:
        logger.warning("⚠️ [Predict] Nenhuma notícia disponível para o usuário %s.", user_id)
        return pd.DataFrame(), non_viewed

    # Obtém as features do cliente e cria DataFrame com uma única linha
    client_feat = get_client_features(user_id, full_df)
    logger.info("👤 [Predict] Features do cliente obtidas para usuário %s.", user_id)
    client_df = pd.DataFrame([client_feat])
    logger.info("📋 [Predict] Colunas do cliente: %s", client_df.columns.tolist())
    
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


def predict_for_user(user_id: str, full_df: pd.DataFrame, model,
                     n: int = 5, score_threshold: float = 15) -> List[str]:
    """
    Realiza a predição e gera recomendações para o usuário.
    
    Args:
        user_id (str): ID do usuário.
        full_df (pd.DataFrame): DataFrame completo de features.
        model: Modelo carregado.
        n (int, optional): Máximo de recomendações.
        score_threshold (float, optional): Score mínimo para recomendação.
    
    Returns:
        List[str]: Lista de IDs das notícias recomendadas.
    """
    final_input, non_viewed = build_model_input(user_id, full_df)
    if final_input.empty:
        logger.info("🙁 [Predict] Nenhum input construído para o usuário %s.", user_id)
        return []
    
    # Realiza a predição
    scores = model.predict(final_input)
    logger.info("🔮 [Predict] Predição realizada para o usuário %s com %d scores.", user_id, len(scores))
    
    # Gera as recomendações
    recommendations = get_predicted_news(scores, non_viewed, n=n, score_threshold=score_threshold)
    logger.info("🎯 [Predict] Recomendações geradas para o usuário %s.", user_id)
    return recommendations


def main():
    logger.info("=== 🚀 [Predict] Iniciando Pipeline de Predição ===")
    pred_file_path = prepare_for_prediction()
    configure_mlflow()
    
    full_df = load_prediction_data(pred_file_path)
    model = load_model_from_mlflow()
    
    user_id = "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297"
    logger.info("=== 🚀 [Predict] Processando predição para o usuário: %s ===", user_id)
    recommendations = predict_for_user(user_id, full_df, model)
    
    if recommendations:
        logger.info("👍 [Predict] Recomendações para o usuário %s: %s", user_id, recommendations)
        print("🔔 Recomendações:")
        for rec in recommendations:
            print(" -", rec)
    else:
        logger.info("😕 [Predict] Nenhuma recomendação gerada para o usuário %s.", user_id)
    
    logger.info("=== ✅ [Predict] Pipeline de Predição Finalizado ===")


if __name__ == "__main__":
    main()
