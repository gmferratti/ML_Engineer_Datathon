import pandas as pd
from .constants import USERS_COLS_TO_EXPLODE, USERS_DTYPES
from src.config import SAMPLE_RATE, COLD_START_THRESHOLD, USERS_DIRECTORY, logger
from .utils import concatenate_csv_files


def preprocess_users() -> pd.DataFrame:
    """
    Pré-processa dados de usuários e retorna o DataFrame processado.

    Returns:
        pd.DataFrame: Dados dos usuários processados.
    """
    logger.info("👥 [Users] Iniciando pré-processamento dos usuários...")
    users_df = concatenate_csv_files(USERS_DIRECTORY)
    logger.info("👥 [Users] Dados carregados: %d linhas (antes da amostragem).", len(users_df))

    users_df = users_df.sample(frac=SAMPLE_RATE, random_state=42)
    logger.info(
        "👥 [Users] Amostragem aplicada (taxa: %.2f). Linhas após amostragem: %d",
        SAMPLE_RATE,
        len(users_df),
    )

    users_df = _process_history_columns(users_df)
    logger.info("👥 [Users] Histórico processado.")

    users_df = users_df.astype(USERS_DTYPES)
    logger.info("👥 [Users] Conversão de tipos realizada.")

    users_df = _process_timestamp(users_df)
    logger.info("👥 [Users] Timestamps processados.")

    users_df = _extract_time_features(users_df)
    logger.info("👥 [Users] Novas features temporais extraídas.")

    users_df["coldStart"] = users_df["historySize"] < COLD_START_THRESHOLD
    logger.info("👥 [Users] Flag 'coldStart' definida (threshold: %d).", COLD_START_THRESHOLD)

    users_df.rename(columns={"history": "pageId"}, inplace=True)
    users_df.drop(columns=["timestampHistory", "timestampHistory_new"], inplace=True)
    logger.info("👥 [Users] Renomeação e remoção de colunas concluídas.")

    users_df = _downcast_columns(users_df)
    logger.info("👥 [Users] Downcast realizado nos dados numéricos.")
    logger.info("👥 [Users] Pré-processamento dos usuários concluído: %d linhas.", len(users_df))

    return users_df


def _process_history_columns(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas de histórico em listas e remove espaços.

    Args:
        users_df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.DataFrame: Dados com histórico processado.
    """
    users_df[USERS_COLS_TO_EXPLODE] = users_df[USERS_COLS_TO_EXPLODE].apply(
        lambda col: col.str.split(",")
    )
    users_df = users_df.explode(USERS_COLS_TO_EXPLODE)
    users_df[USERS_COLS_TO_EXPLODE] = users_df[USERS_COLS_TO_EXPLODE].apply(
        lambda col: col.str.strip()
    )
    return users_df


def _process_timestamp(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte timestamps para datetime e calcula minutos desde o último acesso.

    Args:
        users_df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.DataFrame: Dados com timestamp processado.
    """
    users_df["timestampHistory"] = pd.to_datetime(users_df["timestampHistory"] / 1000, unit="s")
    users_df = users_df.sort_values(by=["userId", "timestampHistory"]).reset_index(drop=True)
    users_df["minutesSinceLastVisit"] = users_df.groupby("userId")["timestampHistory"].diff()
    users_df["minutesSinceLastVisit"] = (
        users_df["minutesSinceLastVisit"].dt.total_seconds().div(60.0).fillna(0).round()
    )
    return users_df


def _extract_time_features(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai novas colunas temporais a partir do timestampHistory.

    Args:
        users_df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.DataFrame: Dados com novas features temporais.
    """
    users_df["timestampHistoryDate"] = users_df["timestampHistory"].dt.date
    users_df["timestampHistoryTime"] = users_df["timestampHistory"].dt.strftime("%H:%M:%S")
    users_df["timestampHistoryWeekday"] = users_df["timestampHistory"].dt.dayofweek
    users_df["timestampHistoryHour"] = users_df["timestampHistory"].dt.hour
    users_df["isWeekend"] = users_df["timestampHistoryWeekday"] >= 5
    users_df["dayPeriod"] = _classify_day_period(users_df)
    return users_df


def _classify_day_period(users_df: pd.DataFrame) -> pd.Series:
    """
    Classifica o período do dia com base na hora.

    Args:
        users_df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.Series: Período do dia.
    """
    return pd.cut(
        users_df["timestampHistoryHour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["dawn", "morning", "afternoon", "night"],
        right=True,
    )


def _downcast_columns(users_df: pd.DataFrame) -> pd.DataFrame:
    """
    Otimiza os tipos de dados numéricos.

    Args:
        users_df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.DataFrame: Dados otimizados.
    """
    users_df["historySize"] = pd.to_numeric(users_df["historySize"], downcast="integer")
    users_df["numberOfClicksHistory"] = pd.to_numeric(
        users_df["numberOfClicksHistory"], downcast="integer"
    )
    users_df["timeOnPageHistory"] = pd.to_numeric(
        users_df["timeOnPageHistory"], downcast="integer"
    )
    users_df["pageVisitsCountHistory"] = pd.to_numeric(
        users_df["pageVisitsCountHistory"], downcast="integer"
    )
    users_df["scrollPercentageHistory"] = pd.to_numeric(
        users_df["scrollPercentageHistory"], downcast="float"
    )
    users_df["minutesSinceLastVisit"] = pd.to_numeric(
        users_df["minutesSinceLastVisit"], downcast="float"
    )
    users_df["timestampHistoryWeekday"] = users_df["timestampHistoryWeekday"].astype("int16")
    users_df["timestampHistoryHour"] = users_df["timestampHistoryHour"].astype("int16")
    return users_df
