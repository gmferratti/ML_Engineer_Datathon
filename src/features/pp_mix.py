"""Módulo que gera e pré-processa as features combinando dados de notícias e usuários."""

import pandas as pd
import numpy as np

from constants import (
    MIX_FEATS_COLS,
    STATE_COLS,
    REGION_COLS,
    THEME_MAIN_COLS,
    THEME_SUB_COLS,
    GAP_COLS,
    FINAL_MIX_FEAT_COLS,
)


def generate_suggested_feats(df_mix: pd.DataFrame,
                             state_df: pd.DataFrame,
                             region_df: pd.DataFrame,
                             tm_df: pd.DataFrame,
                             ts_df: pd.DataFrame) -> pd.DataFrame:
    """
    Gera a tabela final agregando as informações de diferentes dimensões.

    Parâmetros:
        df_mix (DataFrame): Tabela base com as features principais,
            filtrada pelas colunas definidas em FINAL_MIX_FEAT_COLS.
        state_df (DataFrame): Tabela com informações de estado (STATE_COLS),
            agregada por 'userId'.
        region_df (DataFrame): Tabela com informações de região (REGION_COLS),
            agregada por 'userId'.
        tm_df (DataFrame): Tabela com informações de tema principal (THEME_MAIN_COLS),
            agregada por 'userId'.
        ts_df (DataFrame): Tabela com informações de tema secundário (THEME_SUB_COLS),
            agregada por 'userId'.

    Retorna:
        DataFrame: Tabela final com todas as informações agregadas.
    """
    # Filtra a base de features finais pelas colunas necessárias.
    suggested_feats = df_mix[FINAL_MIX_FEAT_COLS]

    # Realiza merges com base nas colunas de chave.
    suggested_feats = suggested_feats.merge(
        state_df, on=["userId", "localState"], how="left"
    )
    suggested_feats = suggested_feats.merge(
        region_df, on=["userId", "localRegion"], how="left"
    )
    suggested_feats = suggested_feats.merge(
        tm_df, on=["userId", "themeMain"], how="left"
    )
    suggested_feats = suggested_feats.merge(
        ts_df, on=["userId", "themeSub"], how="left"
    )
    
    # Remove colunas auxiliares que começam com "count"
    suggested_feats = suggested_feats.drop(columns=[col for col in suggested_feats.columns if col.startswith("count")])
    
    return suggested_feats


def preprocess_mix_feats(df_news: pd.DataFrame,
                         df_users: pd.DataFrame):
    """
    Pré-processa e combina os dataframes de notícias e usuários, 
    criando features temporais, contagens de categorias e separando os dados
    em diferentes DataFrames.

    Parâmetros:
        df_news : pd.DataFrame
            DataFrame com dados das notícias (ex.: 'issuedDate', 'issuedTime', etc.).
        df_users : pd.DataFrame
            DataFrame com o histórico dos usuários (ex.: 'timestampHistoryDate', etc.).

    Retorna:
        tuple:
            (df_mix, gap_df, state_df, region_df, tm_df, ts_df)
            - df_mix: DataFrame principal filtrado pelas colunas em MIX_FEATS_COLS.
            - gap_df: DataFrame com informações do gap temporal.
            - state_df: DataFrame com informações de 'localState'.
            - region_df: DataFrame com informações de 'localRegion'.
            - tm_df: DataFrame com informações de 'themeMain'.
            - ts_df: DataFrame com informações de 'themeSub'.
    """
    df_news, df_users = _process_datetime(df_news, df_users)

    # Combina users e news de acordo com 'pageId' e filtra as colunas.
    df_mix = pd.merge(
        df_users, df_news, on='pageId', how='inner'
    )[MIX_FEATS_COLS]

    # Adiciona colunas de gap temporal.
    df_mix_enriched = _compute_time_gap(df_mix)

    # Adiciona colunas de contagens de categorias.
    df_mix_enriched = _compute_category_counts(df_mix_enriched)

    # Separa em dataframes diferentes.
    gap_df, state_df, region_df, tm_df, ts_df = _split_dataframes(df_mix_enriched)

    return df_mix, gap_df, state_df, region_df, tm_df, ts_df


def _process_datetime(df_news: pd.DataFrame,
                      df_users: pd.DataFrame):
    """
    Converte datas e horários para datetime e cria colunas de timestamps completos.
    """
    # Converte colunas de data em datetime
    df_news['issuedDate'] = pd.to_datetime(
        df_news['issuedDate'], format='%Y-%m-%d'
    )
    df_users['timestampHistoryDate'] = pd.to_datetime(
        df_users['timestampHistoryDate'], format='%Y-%m-%d'
    )

    # Converte colunas de horário em time
    df_news['issuedTime'] = pd.to_datetime(
        df_news['issuedTime'], format='%H:%M:%S', errors='coerce'
    ).dt.time
    df_users['timestampHistoryTime'] = pd.to_datetime(
        df_users['timestampHistoryTime'], format='%H:%M:%S', errors='coerce'
    ).dt.time

    # Cria as colunas de datetime completo
    df_news['issuedDatetime'] = df_news['issuedDate'] + df_news['issuedTime'].apply(
        lambda t: pd.Timedelta(
            hours=t.hour, minutes=t.minute, seconds=t.second
        ) if pd.notnull(t) else pd.Timedelta(0)
    )
    df_users['timestampHistoryDatetime'] = (
        df_users['timestampHistoryDate'] + df_users['timestampHistoryTime'].apply(
            lambda t: pd.Timedelta(
                hours=t.hour, minutes=t.minute, seconds=t.second
            ) if pd.notnull(t) else pd.Timedelta(0)
        )
    )
    return df_news, df_users


def _compute_time_gap(df_mix: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a diferença de tempo entre o histórico do usuário e a publicação da notícia.
    """
    gap = df_mix['timestampHistoryDatetime'] - df_mix['issuedDatetime']
    df_mix['timeGapDays'] = gap.dt.days
    df_mix['timeGapHours'] = gap / pd.Timedelta(hours=1)
    df_mix['timeGapMinutes'] = gap / pd.Timedelta(minutes=1)
    df_mix['timeGapLessThanOneDay'] = df_mix['timeGapHours'] <= 24
    return df_mix


def _compute_category_counts(df_mix: pd.DataFrame,
                             category_columns=None) -> pd.DataFrame:
    """
    Conta quantas notícias o usuário consumiu por categoria (ex.: estado, região),
    cria colunas de quantidade e porcentagem relativa.
    """
    if category_columns is None:
        category_columns = ['localState', 'localRegion', 'themeMain', 'themeSub']
    
    # Conta ocorrências de cada categoria por usuário
    for col in category_columns:
        col_title = col[0].upper() + col[1:]
        count_col = f'count{col_title}User'
        df_mix[count_col] = df_mix.groupby(['userId', col])['pageId'].transform('count')

    df_mix['totalUserNews'] = df_mix.groupby('userId')['pageId'].transform('count')

    # Cria colunas de porcentagem relativa ao total de notícias consumidas
    for col in category_columns:
        col_title = col[0].upper() + col[1:]
        count_col = f'count{col_title}User'
        rel_col = f'rel{col_title}'
        df_mix[rel_col] = df_mix[count_col] / df_mix['totalUserNews']
    
    return df_mix


def _split_dataframes(df_mix: pd.DataFrame):
    """
    Separa o DataFrame principal em múltiplos dataframes baseados em colunas
    de interesse, filtrando valores inválidos (ex.: timeGap negativo).
    """
    gap_df = df_mix[GAP_COLS].copy()
    gap_df = gap_df[gap_df["timeGapDays"] >= 0].reset_index(drop=True)

    state_df = df_mix[STATE_COLS].copy()
    state_df = state_df[state_df["countLocalStateUser"] > 0].reset_index(drop=True)

    region_df = df_mix[REGION_COLS].copy()
    region_df = region_df[region_df["countLocalRegionUser"] > 0].reset_index(drop=True)

    tm_df = df_mix[THEME_MAIN_COLS].copy()
    tm_df = tm_df[tm_df["countThemeMainUser"] > 0].reset_index(drop=True)

    ts_df = df_mix[THEME_SUB_COLS].copy()
    ts_df = ts_df[ts_df["countThemeSubUser"] > 0].reset_index(drop=True)
    
    # Dropando duplicatas
    state_df = state_df.drop_duplicates(subset=["userId", "localState"])
    region_df = region_df.drop_duplicates(subset=["userId", "localRegion"])
    tm_df = tm_df.drop_duplicates(subset=["userId", "themeMain"])
    ts_df = ts_df.drop_duplicates(subset=["userId", "themeSub"])

    return gap_df, state_df, region_df, tm_df, ts_df
