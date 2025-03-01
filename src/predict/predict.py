import os
import pandas as pd
from typing import List
from src.data.data_loader import (
    get_client_features,
    # get_non_viewed_news,
    get_predicted_news,
)
from src.config import DATA_PATH, logger, USE_S3
from src.train.core import load_model_from_mlflow
from storage.io import Storage


def prepare_for_prediction() -> str:
    """
    Lê o dataframe completo de features e o salva na pasta de predição.

    Returns:
        str: Caminho completo do arquivo salvo.
    """
    data_path = DATA_PATH
    logger.info("Utilizando armazenamento: %s", "S3" if USE_S3 else "local")
    storage = Storage(use_s3=USE_S3)
    full_path = os.path.join(data_path, "train", "X_train_full.parquet")
    logger.info("Lendo features: %s", full_path)
    df = storage.read_parquet(full_path)
    pred_dir = os.path.join(data_path, "predict")
    os.makedirs(pred_dir, exist_ok=True)
    save_path = os.path.join(pred_dir, "X_train_full.parquet")
    storage.write_parquet(df, save_path)
    logger.info("Arquivo salvo: %s", save_path)
    return save_path


def predict_for_userId(
    userId: str, full_df: pd.DataFrame, model, n: int = 5, score_threshold: float = 15
) -> List[str]:
    """
    Gera recomendações para um usuário com base no dataframe completo.

    Args:
        userId (str): ID do usuário.
        full_df (pd.DataFrame): DataFrame completo com features.
        model: Modelo treinado.
        n (int, optional): Máximo de recomendações. Default: 5.
        score_threshold (float, optional): Score mínimo. Default: 15.

    Returns:
        List[str]: Lista de IDs das notícias recomendadas.
    """
    seen = full_df.loc[full_df["userId"] == userId, "pageId"].unique()
    non_viewed = full_df[~full_df["pageId"].isin(seen)].copy()
    if non_viewed.empty:
        logger.warning("Nenhuma notícia disponível para recomendar.")
        return []
    client_feat = get_client_features(userId, full_df)
    client_df = pd.DataFrame([client_feat])
    model_input = non_viewed.assign(userId=userId).merge(
        client_df.drop(columns=["userId"]), how="cross", suffixes=("_news", "_user")
    )
    scores = model.predict(model_input)
    return get_predicted_news(scores, non_viewed, n=n, score_threshold=score_threshold)


def main():
    logger.info("Preparando dados para predição...")
    prepare_for_prediction()
    data_path = DATA_PATH
    pred_path = os.path.join(data_path, "predict", "X_train_full.parquet")
    full_df = pd.read_parquet(pred_path)
    model = load_model_from_mlflow()
    userId = "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297"
    logger.info("Predizendo para o usuário: %s", userId)
    recs = predict_for_userId(userId, full_df, model)
    logger.info("Recomendações para %s:", userId)
    if recs:
        for r in recs:
            print(r)
    else:
        logger.info("Nenhuma recomendação gerada.")


if __name__ == "__main__":
    main()
