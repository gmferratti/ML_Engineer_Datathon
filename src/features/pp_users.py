import pandas as pd
from .constants import USERS_COLS_TO_EXPLODE, USERS_DTYPES
from src.config import SAMPLE_RATE, COLD_START_THRESHOLD, USERS_DIRECTORY
from .utils import concatenate_csv_files


def preprocess_users() -> pd.DataFrame:
    """
    Pré-processa dados de usuários.

    Returns:
        pd.DataFrame: Dados dos usuários processados.
    """
    df = concatenate_csv_files(USERS_DIRECTORY)
    df = df.sample(frac=SAMPLE_RATE, random_state=42)
    df = _process_history_columns(df)
    df = df.astype(USERS_DTYPES)
    df = _process_timestamp(df)
    df = _extract_time_features(df)
    df["coldStart"] = df["historySize"] < COLD_START_THRESHOLD
    df.rename(columns={"history": "pageId"}, inplace=True)
    df.drop(columns=["timestampHistory", "timestampHistory_new"], inplace=True)
    return _downcast_columns(df)


def _process_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas de histórico em listas e remove espaços.

    Args:
        df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.DataFrame: Dados com histórico processado.
    """
    df[USERS_COLS_TO_EXPLODE] = df[USERS_COLS_TO_EXPLODE].apply(lambda col: col.str.split(","))
    df = df.explode(USERS_COLS_TO_EXPLODE)
    df[USERS_COLS_TO_EXPLODE] = df[USERS_COLS_TO_EXPLODE].apply(lambda col: col.str.strip())
    return df


def _process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte timestamps para datetime e calcula minutos desde o último acesso.

    Args:
        df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.DataFrame: Dados com timestamp processado.
    """
    df["timestampHistory"] = pd.to_datetime(df["timestampHistory"] / 1000, unit="s")
    df = df.sort_values(by=["userId", "timestampHistory"]).reset_index(drop=True)
    df["minutesSinceLastVisit"] = df.groupby("userId")["timestampHistory"].diff()
    df["minutesSinceLastVisit"] = (
        df["minutesSinceLastVisit"].dt.total_seconds().div(60.0).fillna(0).round()
    )
    return df


def _extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrai novas colunas temporais do timestampHistory.

    Args:
        df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.DataFrame: Dados com novas colunas temporais.
    """
    df["timestampHistoryDate"] = df["timestampHistory"].dt.date
    df["timestampHistoryTime"] = df["timestampHistory"].dt.strftime("%H:%M:%S")
    df["timestampHistoryWeekday"] = df["timestampHistory"].dt.dayofweek
    df["timestampHistoryHour"] = df["timestampHistory"].dt.hour
    df["isWeekend"] = df["timestampHistoryWeekday"] >= 5
    df["dayPeriod"] = _classify_day_period(df)
    return df


def _classify_day_period(df: pd.DataFrame) -> pd.Series:
    """
    Classifica o período do dia com base na hora.

    Args:
        df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.Series: Período do dia.
    """
    return pd.cut(
        df["timestampHistoryHour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["dawn", "morning", "afternoon", "night"],
        right=True,
    )


def _downcast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Faz downcast de colunas numéricas.

    Args:
        df (pd.DataFrame): Dados dos usuários.

    Returns:
        pd.DataFrame: Dados otimizados.
    """
    df["historySize"] = pd.to_numeric(df["historySize"], downcast="integer")
    df["numberOfClicksHistory"] = pd.to_numeric(df["numberOfClicksHistory"], downcast="integer")
    df["timeOnPageHistory"] = pd.to_numeric(df["timeOnPageHistory"], downcast="integer")
    df["pageVisitsCountHistory"] = pd.to_numeric(df["pageVisitsCountHistory"], downcast="integer")
    df["scrollPercentageHistory"] = pd.to_numeric(df["scrollPercentageHistory"], downcast="float")
    df["minutesSinceLastVisit"] = pd.to_numeric(df["minutesSinceLastVisit"], downcast="float")
    df["timestampHistoryWeekday"] = df["timestampHistoryWeekday"].astype("int16")
    df["timestampHistoryHour"] = df["timestampHistoryHour"].astype("int16")
    return df
