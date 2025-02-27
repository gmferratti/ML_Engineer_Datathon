"""Módulo de pré-processamento para geração de coluna TARGET."""

import pandas as pd
import numpy as np

from constants import TARGET_INIT_COLS, TARGET_FINAL_COLS


def preprocess_target(df_users: pd.DataFrame, gap_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera a coluna TARGET para medir o engajamento por par (usuário/página).

    O pipeline inclui:
      1. Merge entre df_users e gap_df.
      2. Cálculo de componentes do score base (clicks, tempo, scroll, recency).
      3. Aplicação de fatores de histórico e gap de tempo.
      4. Padronização do score final via robust scaling (mediana e IQR).

    Args:
        df_users (pd.DataFrame): DataFrame com informações de usuários.
        gap_df (pd.DataFrame): DataFrame com colunas de gap temporal.

    Returns:
        pd.DataFrame: DataFrame contendo as colunas essenciais do TARGET.
    """
    # 1. Mescla DataFrames com base em userId e pageId.
    target_df = df_users.merge(gap_df, on=["userId", "pageId"],how="left")[TARGET_INIT_COLS]

    # 2. Componentes do score base
    clicks_component = target_df["numberOfClicksHistory"]
    time_component = 1.5 * (target_df["timeOnPageHistory"] / 1000)
    scroll_component = target_df["scrollPercentageHistory"]
    recency_penalty = target_df["minutesSinceLastVisit"] / 60

    # Cálculo do score base
    base_score = clicks_component + time_component + scroll_component - recency_penalty

    # Fator de histórico (normalizado pela média ~130)
    history_factor = target_df["historySize"] / 130

    # Fator penalizador para o gap de tempo (quanto menor o gap, maior o score)
    gap_factor = 1 / (1 + target_df["timeGapDays"] / 50)

    # 3. Calcula TARGET combinando componentes e fatores
    target_df["TARGET"] = base_score * history_factor * gap_factor

    # 4. Padroniza TARGET via robust scaling (mediana e IQR)
    median_val = target_df["TARGET"].median()
    iqr_val = target_df["TARGET"].quantile(0.75) - target_df["TARGET"].quantile(0.25)

    if iqr_val == 0:
        target_df["TARGET"] = target_df["TARGET"] - median_val
    else:
        target_df["TARGET"] = (target_df["TARGET"] - median_val) / iqr_val

    # 5. Retém apenas as colunas essenciais
    target_df = target_df[TARGET_FINAL_COLS]

    return target_df
