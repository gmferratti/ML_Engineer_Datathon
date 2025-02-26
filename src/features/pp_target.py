import pandas as pd
import numpy as np
from constants import TARGET_INIT_COLS, TARGET_FINAL_COLS

import pandas as pd
import numpy as np
from constants import TARGET_INIT_COLS, TARGET_FINAL_COLS

def preprocess_target(
    df_users: pd.DataFrame,
    gap_df: pd.DataFrame
) -> pd.DataFrame:
    """Gera a coluna TARGET para medir o engajamento por par usuário/página."""
    
    # Mescla os DataFrames com base em userId e pageId e seleciona as colunas iniciais necessárias
    target_df = df_users.merge(gap_df, on=["userId", "pageId"])[TARGET_INIT_COLS]

    # Componentes do score base
    clicks_component = target_df['numberOfClicksHistory']
    time_component = 1.5 * (target_df['timeOnPageHistory'] / 1000)
    scroll_component = target_df['scrollPercentageHistory']
    recency_penalty = target_df['minutesSinceLastVisit'] / 60

    # Cálculo do score base
    base_score = clicks_component + time_component + scroll_component - recency_penalty

    # Fator para recompensar usuários com maior historySize (normalizado pela média, 130)
    history_factor = target_df['historySize'] / 130

    # Fator penalizador para o gap de tempo: quanto menor o gap, maior o score
    gap_factor = 1 / (1 + target_df['timeGapDays'] / 50)

    # Calcula o TARGET combinando os componentes, a recompensa por historySize e a penalização pelo timeGapDays
    target_df['TARGET'] = base_score * history_factor * gap_factor

    # Padroniza TARGET utilizando robust scaling (subtrai a mediana e divide pelo IQR)
    median_val = target_df['TARGET'].median()
    iqr_val = target_df['TARGET'].quantile(0.75) - target_df['TARGET'].quantile(0.25)

    if iqr_val == 0:
        target_df['TARGET'] = target_df['TARGET'] - median_val
    else:
        target_df['TARGET'] = (target_df['TARGET'] - median_val) / iqr_val

    # Mantém apenas as colunas essenciais do dataframe de target
    target_df = target_df[TARGET_FINAL_COLS]

    return target_df
