"""Módulo de pré-processamento para geração de coluna TARGET."""

import pandas as pd
import numpy as np

from constants import TARGET_INIT_COLS, TARGET_FINAL_COLS, DEFAULT_TARGET_VALUES
from config import SCALING_RANGE, logger


def preprocess_target(df_users: pd.DataFrame, gap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera a coluna TARGET para medir o engajamento por par (usuário/página) com maior variância.

    Pipeline:
        1. Merge entre df_users e gap_df (chaves: userId, pageId).
        2. Tratamento de NAs e cálculo de um score base com pesos ajustados.
        3. Aplicação de fatores de histórico e gap de tempo.
        4. Transformação logarítmica (log1p) para "espalhar" a distribuição.
        5. Escalonamento Min-Max para [0, SCALING_RANGE].
        6. Conversão final para inteiro (round).

    Args:
        df_users (pd.DataFrame): DataFrame com informações de usuários.
        gap_df (pd.DataFrame): DataFrame com colunas de gap temporal.

    Returns:
        pd.DataFrame: DataFrame contendo colunas ["userId", "pageId", "TARGET"].
    """
    logger.info("Iniciando preprocessamento do target...")

    # 1. Merge entre df_users e gap_df
    logger.info("Realizando merge dos dataframes: df_users (%s linhas) e gap_df (%s linhas).",
                len(df_users), len(gap_df))
    target_df = df_users.merge(gap_df, on=["userId", "pageId"], how="left")[TARGET_INIT_COLS].copy()
    logger.info("Merge concluído. DataFrame resultante com %s linhas e %s colunas.",
                target_df.shape[0], target_df.shape[1])

    # 2. Tratamento de valores ausentes
    logger.info("Preenchendo valores ausentes com DEFAULT_TARGET_VALUES: %s", DEFAULT_TARGET_VALUES)
    for col, default in DEFAULT_TARGET_VALUES.items():
        if col in target_df.columns:
            target_df[col] = target_df[col].fillna(default)

    # 3. Cálculo do score base (com pesos ajustados)
    logger.info("Calculando o score base com pesos definidos.")
    clicks_component = 1.0 * target_df["numberOfClicksHistory"]
    time_component = 2.0 * (target_df["timeOnPageHistory"] / 1000)   # aumentar peso do tempo
    scroll_component = 1.5 * target_df["scrollPercentageHistory"]   # aumentar peso do scroll
    recency_penalty = 0.5 * (target_df["minutesSinceLastVisit"] / 60)

    base_score = (clicks_component + time_component + scroll_component) - recency_penalty

    # Fatores de histórico e gap
    history_factor = target_df["historySize"] / 130
    gap_factor = 1 / (1 + target_df["timeGapDays"] / 50)

    raw_score = base_score * history_factor * gap_factor

    # Forçar mínimo de 0 (evitar problemas no log)
    raw_score = np.maximum(raw_score, 0)

    # 4. Transformação logarítmica
    logger.info("Aplicando transformação logarítmica (log1p).")
    log_score = np.log1p(raw_score)

    # 5. Escalonamento MinMax para [0, SCALING_RANGE]
    min_val, max_val = log_score.min(), log_score.max()
    logger.info("Valores mínimo e máximo antes do scaling: min=%.4f, max=%.4f", min_val, max_val)

    if max_val == min_val:
        # Caso extremo: todos os valores são iguais
        scaled_score = log_score - min_val
        logger.warning("Todos os valores de score são idênticos; escalonamento resultará em 0.")
    else:
        scaled_score = (log_score - min_val) / (max_val - min_val)

    scaled_score *= SCALING_RANGE

    # Arredondar para inteiro
    final_score = scaled_score.round().astype(int)

    # 6. Atribuir resultado ao DataFrame
    target_df["TARGET"] = final_score

    # 7. Retornar somente colunas finais
    target_df = target_df[TARGET_FINAL_COLS]

    logger.info("Preprocessamento concluído. Retornando DataFrame final com shape %s.",
                target_df.shape)
    return target_df