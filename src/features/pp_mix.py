import pandas as pd
from features.constants import (
    MIX_FEATS_COLS,
    STATE_COLS,
    REGION_COLS,
    THEME_MAIN_COLS,
    THEME_SUB_COLS,
    GAP_COLS,
    FINAL_MIX_FEAT_COLS,
)
from src.config import logger


def generate_suggested_feats(
    df_mix: pd.DataFrame,
    state_df: pd.DataFrame,
    region_df: pd.DataFrame,
    tm_df: pd.DataFrame,
    ts_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Gera a tabela final agregando informações de várias dimensões.

    Args:
        df_mix (pd.DataFrame): Base com features principais.
        state_df (pd.DataFrame): Dados de estado.
        region_df (pd.DataFrame): Dados de região.
        tm_df (pd.DataFrame): Dados de tema principal.
        ts_df (pd.DataFrame): Dados de tema secundário.

    Returns:
        pd.DataFrame: Tabela final agregada.
    """
    logger.info("📐 [Mix] Gerando suggested_feats...")
    suggested = df_mix[FINAL_MIX_FEAT_COLS]
    suggested = suggested.merge(state_df, on=["userId", "localState"], how="left")
    suggested = suggested.merge(region_df, on=["userId", "localRegion"], how="left")
    suggested = suggested.merge(tm_df, on=["userId", "themeMain"], how="left")
    suggested = suggested.merge(ts_df, on=["userId", "themeSub"], how="left")
    cols = [col for col in suggested.columns if col.startswith("count")]
    logger.info("📐 [Mix] Removendo colunas de contagem...")
    return suggested.drop(columns=cols)


def preprocess_mix_feats(df_news: pd.DataFrame, df_users: pd.DataFrame):
    """
    Pré-processa e combina os dataframes de notícias e usuários.

    Args:
        df_news (pd.DataFrame): Dados das notícias.
        df_users (pd.DataFrame): Dados dos usuários.

    Returns:
        tuple: (df_mix, gap_df, state_df, region_df, tm_df, ts_df)
    """
    logger.info("🔀 [Mix] Iniciando pré-processamento do mix_feats...")
    df_news, df_users = _process_datetime(df_news, df_users)
    df_mix = pd.merge(df_users, df_news, on="pageId", how="inner")[MIX_FEATS_COLS]
    df_mix = _compute_time_gap(df_mix)
    df_mix = _compute_category_counts(df_mix)
    logger.info("🔀 [Mix] Finalizando pré-processamento do mix_feats...")
    return _split_dataframes(df_mix)


def _process_datetime(df_news: pd.DataFrame, df_users: pd.DataFrame):
    """
    Converte datas e horários para datetime e cria timestamps.

    Args:
        df_news (pd.DataFrame): Dados das notícias.
        df_users (pd.DataFrame): Dados dos usuários.

    Returns:
        tuple: (df_news, df_users) com novas colunas.
    """
    logger.info("🕒 [Mix] Processando datas e horários...")
    df_news["issuedDate"] = pd.to_datetime(df_news["issuedDate"], format="%Y-%m-%d")
    df_users["timestampHistoryDate"] = pd.to_datetime(
        df_users["timestampHistoryDate"], format="%Y-%m-%d"
    )
    df_news["issuedTime"] = pd.to_datetime(
        df_news["issuedTime"], format="%H:%M:%S", errors="coerce"
    ).dt.time
    df_users["timestampHistoryTime"] = pd.to_datetime(
        df_users["timestampHistoryTime"], format="%H:%M:%S", errors="coerce"
    ).dt.time
    df_news["issuedDatetime"] = df_news["issuedDate"] + df_news["issuedTime"].apply(
        lambda t: (
            pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
            if pd.notnull(t)
            else pd.Timedelta(0)
        )
    )
    df_users["timestampHistoryDatetime"] = df_users["timestampHistoryDate"] + df_users[
        "timestampHistoryTime"
    ].apply(
        lambda t: (
            pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second)
            if pd.notnull(t)
            else pd.Timedelta(0)
        )
    )
    logger.info("🕒 [Mix] Datas e horários processados.")
    return df_news, df_users


def _compute_time_gap(df_mix: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a diferença de tempo entre publicação e consumo.

    Args:
        df_mix (pd.DataFrame): Dados combinados.

    Returns:
        pd.DataFrame: Dados com colunas de gap temporal.
    """
    logger.info("⏱️ [Mix] Calculando gap temporal...")
    gap = df_mix["timestampHistoryDatetime"] - df_mix["issuedDatetime"]
    df_mix["timeGapDays"] = gap.dt.days
    df_mix["timeGapHours"] = gap / pd.Timedelta(hours=1)
    df_mix["timeGapMinutes"] = gap / pd.Timedelta(minutes=1)
    df_mix["timeGapLessThanOneDay"] = df_mix["timeGapHours"] <= 24
    logger.info("⏱️ [Mix] Gap temporal calculado.")
    return df_mix


def _compute_category_counts(df_mix: pd.DataFrame, category_columns=None) -> pd.DataFrame:
    """
    Conta notícias por categoria e cria colunas com proporção.

    Args:
        df_mix (pd.DataFrame): Dados combinados.
        category_columns (list, optional): Colunas de categoria.

    Returns:
        pd.DataFrame: Dados com contagens e proporções.
    """
    logger.info("📊 [Mix] Calculando contagens por categoria...")
    if category_columns is None:
        category_columns = ["localState", "localRegion", "themeMain", "themeSub"]
    for col in category_columns:
        title = col[0].upper() + col[1:]
        count_col = f"count{title}User"
        df_mix[count_col] = df_mix.groupby(["userId", col])["pageId"].transform("count")
    df_mix["totalUserNews"] = df_mix.groupby("userId")["pageId"].transform("count")
    for col in category_columns:
        title = col[0].upper() + col[1:]
        count_col = f"count{title}User"
        rel_col = f"rel{title}"
        df_mix[rel_col] = df_mix[count_col] / df_mix["totalUserNews"]
    logger.info("📊 [Mix] Contagens por categoria calculadas.")
    return df_mix


def _split_dataframes(df_mix: pd.DataFrame):
    """
    Separa o dataframe em subconjuntos por dimensão.

    Args:
        df_mix (pd.DataFrame): Dados enriquecidos.

    Returns:
        tuple: (df_mix, gap_df, state_df, region_df, tm_df, ts_df)
    """
    logger.info("🔀 [Mix] Separando subconjuntos...")
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
    state_df = state_df.drop_duplicates(subset=["userId", "localState"])
    region_df = region_df.drop_duplicates(subset=["userId", "localRegion"])
    tm_df = tm_df.drop_duplicates(subset=["userId", "themeMain"])
    ts_df = ts_df.drop_duplicates(subset=["userId", "themeSub"])
    logger.info("🔀 [Mix] Subconjuntos separados.")
    return df_mix, gap_df, state_df, region_df, tm_df, ts_df
