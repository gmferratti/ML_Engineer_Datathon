import pandas as pd
import numpy as np
from .constants import TARGET_INIT_COLS, TARGET_FINAL_COLS, DEFAULT_TARGET_VALUES
from src.config import SCALING_RANGE, logger


def preprocess_target(df_users: pd.DataFrame, gap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocessa o target.

    Gera a coluna TARGET para medir o engajamento, combinando
     dados dos usuários com o gap temporal.

    Args:
        df_users (pd.DataFrame): Dados dos usuários.
        gap_df (pd.DataFrame): Dados do gap temporal.

    Returns:
        pd.DataFrame: DataFrame contendo as colunas ["userId", "pageId", "TARGET"].
    """
    logger.info("🎯 [Target] Iniciando pré-processamento do target...")
    logger.info(
        "🎯 [Target] Merge: df_users (%d linhas) + gap_df (%d linhas).",
        len(df_users),
        len(gap_df),
    )

    target_df = df_users.merge(gap_df, on=["userId", "pageId"], how="left")[
        TARGET_INIT_COLS
    ].copy()
    logger.info("🎯 [Target] Merge concluído. Shape: %s", target_df.shape)

    # Preenche valores padrão para colunas ausentes
    for col, default in DEFAULT_TARGET_VALUES.items():
        if col in target_df.columns:
            target_df[col] = target_df[col].fillna(default)

    logger.info("🎯 [Target] Calculando score base...")
    clicks = 1.0 * target_df["numberOfClicksHistory"]
    time_comp = 2.0 * (target_df["timeOnPageHistory"] / 1000)
    scroll = 1.5 * target_df["scrollPercentageHistory"]
    recency = 0.5 * (target_df["minutesSinceLastVisit"] / 60)
    base_score = clicks + time_comp + scroll - recency

    history_factor = target_df["historySize"] / 130
    gap_factor = 1 / (1 + target_df["timeGapDays"] / 50)
    raw_score = base_score * history_factor * gap_factor
    raw_score = np.maximum(raw_score, 0)

    logger.info("🎯 [Target] Aplicando log1p ao score...")
    log_score = np.log1p(raw_score)
    min_val, max_val = log_score.min(), log_score.max()
    logger.info("🎯 [Target] Score: min=%.4f, max=%.4f", min_val, max_val)

    if max_val == min_val:
        scaled = log_score - min_val
        logger.warning("⚠️ [Target] Scores idênticos; escala definida como 0.")
    else:
        scaled = (log_score - min_val) / (max_val - min_val)

    scaled *= SCALING_RANGE
    final_score = scaled.round().astype(int)
    target_df["TARGET"] = final_score

    target_df = target_df[TARGET_FINAL_COLS]
    logger.info("🎯 [Target] Pré-processamento finalizado. Shape final: %s", target_df.shape)
    return target_df
