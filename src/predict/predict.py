import os
import pandas as pd
from typing import List
from src.data.data_loader import get_client_features, get_predicted_news
from src.config import DATA_PATH, logger, USE_S3, configure_mlflow
from src.train.core import load_model_from_mlflow
from storage.io import Storage


def prepare_for_prediction() -> str:
    """
    Lê o dataframe completo de features e o salva na pasta de predição.

    Returns:
        str: Caminho completo do arquivo salvo.
    """
    storage = Storage(use_s3=USE_S3)
    full_path = os.path.join(DATA_PATH, "train", "X_train_full.parquet")
    logger.info("Lendo features de %s", full_path)
    df = storage.read_parquet(full_path)
    pred_dir = os.path.join(DATA_PATH, "predict")
    if not os.path.exists(pred_dir):
        os.makedirs(pred_dir, exist_ok=True)
    save_path = os.path.join(pred_dir, "X_train_full.parquet")
    storage.write_parquet(df, save_path)
    logger.info("Arquivo salvo em %s", save_path)
    return save_path


def predict_for_userId(userId: str, full_df: pd.DataFrame, model,
                       n: int = 5, score_threshold: float = 15) -> List[str]:
    """
    Gera recomendações para um usuário com base no dataframe completo.

    Args:
        userId (str): ID do usuário.
        full_df (pd.DataFrame): DataFrame completo com features.
        model: Modelo treinado.
        n (int, optional): Máximo de recomendações.
        score_threshold (float, optional): Score mínimo.

    Returns:
        List[str]: IDs das notícias recomendadas.
    """
    seen = full_df.loc[full_df["userId"] == userId, "pageId"].unique()
    non_viewed = full_df[~full_df["pageId"].isin(seen)].copy()
    if non_viewed.empty:
        logger.warning("Nenhuma notícia disponível para recomendar.")
        return []
    client_feat = get_client_features(userId, full_df)
    client_df = pd.DataFrame([client_feat])
    model_input = non_viewed.assign(userId=userId).merge(
        client_df.drop(columns=["userId"]), how="cross")
    scores = model.predict(model_input)
    return get_predicted_news(scores, non_viewed, n=n,
                              score_threshold=score_threshold)


def main():
    logger.info("Preparando dados para predição...")
    prepare_for_prediction()
    configure_mlflow()
    pred_path = os.path.join(DATA_PATH, "predict", "X_train_full.parquet")
    storage = Storage(use_s3=USE_S3)
    full_df = storage.read_parquet(pred_path)
    model = load_model_from_mlflow()
    userId = "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297"
    logger.info("Predizendo para o usuário: %s", userId)
    recs = predict_for_userId(userId, full_df, model)
    logger.info("Recomendações para %s: %s", userId, recs)
    if recs:
        for r in recs:
            print(r)
    else:
        logger.info("Nenhuma recomendação gerada.")


if __name__ == "__main__":
    main()
