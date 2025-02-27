"""Módulo de pré-processamento para dados de usuários."""

import pandas as pd

from constants import (
    USERS_COLS_TO_EXPLODE,
    USERS_DTYPES,
)
from feat_settings import (
    SAMPLE_RATE,
    COLD_START_THRESHOLD,
    USERS_N_CSV_FILES,
    USERS_TEMP_PATH,
)
from utils import concatenate_csv_to_df


def preprocess_users() -> pd.DataFrame:
    """
    Realiza o pré-processamento dos dados dos usuários:
      1. Concatena múltiplos CSVs.
      2. Faz amostragem (sampling).
      3. 'Explode' colunas de histórico, converte para lista e remove espaços.
      4. Converte colunas para dtypes apropriados.
      5. Processa timestamps (datetime) e ordena por usuário/data.
      6. Cria variáveis derivadas (minutos desde último acesso, cold start).
      7. Faz downcasting das colunas numéricas para economia de memória.

    Returns:
        pd.DataFrame: DataFrame processado com colunas transformadas.
    """
    # 1. Concatena CSVs
    df_users = concatenate_csv_to_df(USERS_TEMP_PATH, USERS_N_CSV_FILES)

    # 2. Faz sampling dos dados
    df_users = df_users.sample(frac=SAMPLE_RATE, random_state=42)

    # 3. Processa colunas de histórico
    df_users = _process_history_columns(df_users)

    # 4. Converte colunas para tipos apropriados
    df_users = df_users.astype(USERS_DTYPES)

    # 5. Converte timestamp e ordena por usuário/data
    df_users = _process_timestamp(df_users)

    # 6. Extrai variáveis temporais derivadas
    df_users = _extract_time_features(df_users)

    # Cria indicador de cold start
    df_users["coldStart"] = df_users["historySize"] < COLD_START_THRESHOLD

    # Renomeia a coluna de chave secundária
    df_users.rename(columns={"history": "pageId"}, inplace=True)

    # Remove colunas desnecessárias
    df_users.drop(columns=["timestampHistory", "timestampHistory_new"], inplace=True)

    # 7. Downcast de colunas numéricas
    df_users = _downcast_columns(df_users)

    return df_users


def _process_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte colunas de histórico (string) em listas, 'explode' e remove espaços.

    Args:
        df (pd.DataFrame): DataFrame de usuários.

    Returns:
        pd.DataFrame: DataFrame com colunas de histórico explodidas.
    """
    df[USERS_COLS_TO_EXPLODE] = df[USERS_COLS_TO_EXPLODE].apply(
        lambda col: col.str.split(",")
    )
    df = df.explode(USERS_COLS_TO_EXPLODE)
    df[USERS_COLS_TO_EXPLODE] = df[USERS_COLS_TO_EXPLODE].apply(
        lambda col: col.str.strip()
    )
    return df


def _process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """
    Converte timestamps (millis) para datetime e ordena por usuário e data/hora.
    Calcula 'minutesSinceLastVisit' para cada usuário.

    Args:
        df (pd.DataFrame): DataFrame de usuários.

    Returns:
        pd.DataFrame: DataFrame com timestamps em formato datetime,
                      ordenado por userId e timestampHistory.
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
    Extrai colunas derivadas de informação temporal do timestampHistory.

    Args:
        df (pd.DataFrame): DataFrame de usuários.

    Returns:
        pd.DataFrame: DataFrame com novas colunas temporais.
    """
    df["timestampHistoryDate"] = df["timestampHistory"].dt.date
    df["timestampHistoryTime"] = df["timestampHistory"].dt.strftime("%H:%M:%S")
    df["timestampHistoryWeekday"] = df["timestampHistory"].dt.dayofweek
    df["timestampHistoryHour"] = df["timestampHistory"].dt.hour

    # Marca se é fim de semana
    df["isWeekend"] = df["timestampHistoryWeekday"] >= 5

    # Classifica o período do dia
    df["dayPeriod"] = _classify_day_period(df)
    return df


def _classify_day_period(df: pd.DataFrame) -> pd.Series:
    """
    Classifica o período do dia com base na hora.

    Args:
        df (pd.DataFrame): DataFrame de usuários.

    Returns:
        pd.Series: Series com o período do dia (dawn, morning, afternoon, night).
    """
    return pd.cut(
        df["timestampHistoryHour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["dawn", "morning", "afternoon", "night"],
        right=True
    )


def _downcast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Faz downcast das colunas numéricas para reduzir uso de memória.

    Args:
        df (pd.DataFrame): DataFrame de usuários.

    Returns:
        pd.DataFrame: DataFrame com colunas numéricas downcasted.
    """
    df["historySize"] = pd.to_numeric(df["historySize"], downcast="integer")
    df["numberOfClicksHistory"] = pd.to_numeric(
        df["numberOfClicksHistory"], downcast="integer"
    )
    df["timeOnPageHistory"] = pd.to_numeric(
        df["timeOnPageHistory"], downcast="integer"
    )
    df["pageVisitsCountHistory"] = pd.to_numeric(
        df["pageVisitsCountHistory"], downcast="integer"
    )
    df["scrollPercentageHistory"] = pd.to_numeric(
        df["scrollPercentageHistory"], downcast="float"
    )
    df["minutesSinceLastVisit"] = pd.to_numeric(
        df["minutesSinceLastVisit"], downcast="float"
    )
    df["timestampHistoryWeekday"] = df["timestampHistoryWeekday"].astype("int16")
    df["timestampHistoryHour"] = df["timestampHistoryHour"].astype("int16")

    return df
