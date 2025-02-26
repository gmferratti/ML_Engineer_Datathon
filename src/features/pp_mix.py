import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif

from constants import (
    MIX_FEATS_COLS,
    STATE_COLS,
    REGION_COLS,
    THEME_MAIN_COLS,
    THEME_SUB_COLS,
    GAP_COLS,
    FINAL_MIX_FEAT_COLS,
)

def generate_suggested_feats(
    df_mix,
    state_df,
    region_df,
    tm_df,
    ts_df,
):
    """
    Gera a tabela final agregando as informações de diferentes dimensões.

    Parâmetros:
        df_mix (DataFrame): Tabela base com as features principais, filtrada pelas colunas definidas em FINAL_MIX_FEAT_COLS.
        state_df (DataFrame): Tabela com informações de estado (STATE_COLS), agregada por 'userId'.
        region_df (DataFrame): Tabela com informações de região (REGION_COLS), agregada por 'userId'.
        tm_df (DataFrame): Tabela com informações de tema principal (THEME_MAIN_COLS), agregada por 'userId'.
        ts_df (DataFrame): Tabela com informações de tema secundário (THEME_SUB_COLS), agregada por 'userId'.

    Retorna:
        DataFrame: Tabela final com todas as informações agregadas.
    """
    # Filtra a base de features finais para garantir que somente as colunas necessárias sejam utilizadas.
    sug_feats = df_mix[FINAL_MIX_FEAT_COLS].copy()

    # Agrega as dimensões por 'userId'["userId", "pageId"]
    sug_feats = sug_feats.merge(state_df, on=["userId","localState"], how="left")
    sug_feats = sug_feats.merge(region_df, on=["userId","localRegion"], how="left")
    sug_feats = sug_feats.merge(tm_df, on=["userId","themeMain"], how="left")
    sug_feats = sug_feats.merge(ts_df, on=["userId","themeSub"], how="left")

    return sug_feats

def preprocess_mix_feats(df_news: pd.DataFrame, df_users: pd.DataFrame):
    """
    Pré-processa e combina os dataframes de notícias e usuários, criando features temporais,
    contagens de categorias e separando os dados em diferentes DataFrames.

    Parâmetros:
    -----------
    df_news : pd.DataFrame
        DataFrame com dados das notícias (ex.: 'issuedDate', 'issuedTime', 'localState', etc.).
    df_users : pd.DataFrame
        DataFrame com o histórico dos usuários (ex.: 'timestampHistoryDate', 'timestampHistoryTime', 'userId', etc.).

    Retorna:
    --------
    tuple:
        gap_df, state_df, region_df, tm_df, ts_df
        - gap_df: DataFrame com informações do gap temporal.
        - state_df: DataFrame com informações de 'localState'.
        - region_df: DataFrame com informações de 'localRegion'.
        - tm_df: DataFrame com informações de 'themeMain'.
        - ts_df: DataFrame com informações de 'themeSub'.
    """
    df_news, df_users = _process_datetime(df_news, df_users)
    df_mix = pd.merge(df_users, df_news, on='pageId', how='inner')[MIX_FEATS_COLS]

    df_mix_enriched = _compute_time_gap(df_mix)
    
    df_mix_enriched = _compute_category_counts(df_mix_enriched)
    gap_df, state_df, region_df, tm_df, ts_df = _split_dataframes(df_mix_enriched)
    
    return df_mix, gap_df, state_df, region_df, tm_df, ts_df

def _process_datetime(df_news: pd.DataFrame, df_users: pd.DataFrame):
    """Converte datas e horários e cria os timestamps completos."""
    df_news['issuedDate'] = pd.to_datetime(df_news['issuedDate'], format='%Y-%m-%d')
    df_users['timestampHistoryDate'] = pd.to_datetime(df_users['timestampHistoryDate'], format='%Y-%m-%d')
    
    df_news['issuedTime'] = pd.to_datetime(
        df_news['issuedTime'], format='%H:%M:%S', errors='coerce'
    ).dt.time
    df_users['timestampHistoryTime'] = pd.to_datetime(
        df_users['timestampHistoryTime'], format='%H:%M:%S', errors='coerce'
    ).dt.time

    df_news['issuedDatetime'] = df_news['issuedDate'] + df_news['issuedTime'].apply(
        lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second) if pd.notnull(t) else pd.Timedelta(0)
    )
    df_users['timestampHistoryDatetime'] = df_users['timestampHistoryDate'] + df_users['timestampHistoryTime'].apply(
        lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second) if pd.notnull(t) else pd.Timedelta(0)
    )
    return df_news, df_users

def _compute_time_gap(df_mix_enriched: pd.DataFrame):
    gap = df_mix_enriched['timestampHistoryDatetime'] - df_mix_enriched['issuedDatetime']
    df_mix_enriched['timeGapDays'] = gap.dt.days
    df_mix_enriched['timeGapHours'] = gap / pd.Timedelta(hours=1)
    df_mix_enriched['timeGapMinutes'] = gap / pd.Timedelta(minutes=1)
    df_mix_enriched['timeGapLessThanOneDay'] = df_mix_enriched['timeGapHours'] <= 24
    return df_mix_enriched

def _compute_category_counts(df_mix_enriched: pd.DataFrame, category_columns=None):
    if category_columns is None:
        category_columns = ['localState', 'localRegion', 'themeMain', 'themeSub']
    for col in category_columns:
        col_title = col[0].upper() + col[1:]
        count_col = f'count{col_title}User'
        df_mix_enriched[count_col] = df_mix_enriched.groupby(['userId', col])['pageId'].transform('count')
    df_mix_enriched['totalUserNews'] = df_mix_enriched.groupby('userId')['pageId'].transform('count')
    for col in category_columns:
        col_title = col[0].upper() + col[1:]
        count_col = f'count{col_title}User'
        rel_col = f'rel{col_title}'
        df_mix_enriched[rel_col] = df_mix_enriched[count_col] / df_mix_enriched['totalUserNews']
    return df_mix_enriched

def _split_dataframes(df_mix_enriched: pd.DataFrame):
    gap_df = df_mix_enriched[GAP_COLS].copy()
    gap_df = gap_df[gap_df["timeGapDays"] >= 0].reset_index(drop=True)
    
    state_df = df_mix_enriched[STATE_COLS].copy()
    state_df = state_df[state_df["countLocalStateUser"] > 0].reset_index(drop=True)
    
    region_df = df_mix_enriched[REGION_COLS].copy()
    region_df = region_df[region_df["countLocalRegionUser"] > 0].reset_index(drop=True)
    
    tm_df = df_mix_enriched[THEME_MAIN_COLS].copy()
    tm_df = tm_df[tm_df["countThemeMainUser"] > 0].reset_index(drop=True)
    
    ts_df = df_mix_enriched[THEME_SUB_COLS].copy()
    ts_df = ts_df[ts_df["countThemeSubUser"] > 0].reset_index(drop=True)
    
    return gap_df, state_df, region_df, tm_df, ts_df

#TODO: ver melhor local para esta função. Fiz aqui porque já estava no jeito.
def _get_unread_news_for_user(news: pd.DataFrame, users: pd.DataFrame, userId: str) -> pd.DataFrame:
    """
    Retorna um DataFrame com as notícias não lidas para um único usuário.

    Parâmetros:
    -----------
    news : pd.DataFrame
        DataFrame com as notícias disponíveis.
    users : pd.DataFrame
        DataFrame com o histórico completo dos usuários.
    userId : str
        Identificador do usuário para o qual se deseja obter as notícias não lidas.

    Retorna:
    --------
    pd.DataFrame
        DataFrame com as notícias que o usuário ainda não leu, com a coluna 'userId' adicionada.
    """
    read_pages = users.loc[users['userId'] == userId, 'pageId'].unique()
    unread = news[~news['pageId'].isin(read_pages)].copy()
    unread['userId'] = userId
    unread = unread[['userId', 'pageId']].reset_index(drop=True)
    return unread

# Exemplo de chamada na API para um único usuário:
#unread_news_for_user = _get_unread_news_for_user(news, users, userId='ffe133162533bd67689c667be6c302b7342f8a682d28d7')