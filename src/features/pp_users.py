"""Preprocessing for users features."""

import pandas as pd

from .constants import (
    cold_start_threshold,
    users_cols_to_explode,
    users_dtypes,
    users_num_csv_files,
    users_template_path,
)
from .utils import concatenate_csv_to_df


def preprocess_users() -> pd.DataFrame:
    """
    Realiza o pré-processamento dos dados dos usuários:
    - Concatena CSVs.
    - Explode colunas de histórico.
    - Converte colunas para tipos apropriados.
    - Processa informações temporais.
    - Cria variáveis derivadas (ex: minutos desde o último acesso, flag de cold start).
    - Realiza downcasting das colunas.
    """

    # Concatena CSVs
    df_users = concatenate_csv_to_df(users_template_path, users_num_csv_files)

    # Processa colunas de histórico (explode e remove espaços)
    df_users = _process_history_columns(df_users)

    # Converte colunas iniciais para tipos apropriados
    df_users = df_users.astype(users_dtypes)

    # Converte timestamp e ordena por usuário e data
    df_users = _process_timestamp(df_users)

    # Cria variáveis temporais derivadas
    df_users = _extract_time_features(df_users)

    # Cria indicador de fim de semana
    df_users["isWeekend"] = df_users["timestampHistoryWeekday"] >= 5

    # Classifica os períodos do dia
    df_users["dayPeriod"] = _classify_day_period(df_users)

    # Cria indicador de cold start
    df_users["coldStart"] = df_users["historySize"] < cold_start_threshold

    # Renomeia a coluna de chave secundária
    df_users.rename(columns={"history": "historyId"}, inplace=True)

    # Remove colunas desnecessárias
    df_users.drop(columns=["timestampHistory", "timestampHistory_new"], inplace=True)

    # Realiza o downcasting das colunas numéricas
    df_users = _downcast_columns(df_users)

    return df_users


def _process_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas de histórico de string para lista, explode e remove espaços."""
    # Transforma colunas de histórico de string para lista
    df[users_cols_to_explode] = df[users_cols_to_explode].apply(lambda col: col.str.split(","))

    # Explode o dataframe e remove espaços das strings
    df = df.explode(users_cols_to_explode)
    df[users_cols_to_explode] = df[users_cols_to_explode].apply(lambda col: col.str.strip())

    return df


def _process_timestamp(df: pd.DataFrame) -> pd.DataFrame:
    """Converte timestamp para datetime e ordena por usuário e data/hora."""
    df["timestampHistory"] = pd.to_datetime(df["timestampHistory"] / 1000, unit="s")
    df = df.sort_values(by=["userId", "timestampHistory"]).reset_index(drop=True)

    # Calcula diferença em minutos desde o último acesso
    df["minutesSinceLastVisit"] = (
        df.groupby("userId")["timestampHistory"].diff().dt.total_seconds() / 60.0
    )
    df["minutesSinceLastVisit"] = df["minutesSinceLastVisit"].fillna(0).round()

    return df


def _extract_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Extrai informações temporais do timestamp."""
    df["timestampHistoryDate"] = df["timestampHistory"].dt.date
    df["timestampHistoryTime"] = df["timestampHistory"].dt.strftime("%H:%M")
    df["timestampHistoryWeekday"] = df["timestampHistory"].dt.dayofweek
    df["timestampHistoryHour"] = df["timestampHistory"].dt.hour

    return df


def _classify_day_period(df: pd.DataFrame) -> pd.Series:
    """Classifica o período do dia com base na hora."""
    return pd.cut(
        df["timestampHistoryHour"],
        bins=[-1, 5, 11, 17, 23],
        labels=["dawn", "morning", "afternoon", "night"],
        right=True,
    )


def _downcast_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Faz o downcast das colunas numéricas para reduzir uso de memória."""
    df["historySize"] = pd.to_numeric(df["historySize"], downcast="integer")
    df["numberOfClicksHistory"] = pd.to_numeric(df["numberOfClicksHistory"], downcast="integer")
    df["timeOnPageHistory"] = pd.to_numeric(df["timeOnPageHistory"], downcast="integer")
    df["pageVisitsCountHistory"] = pd.to_numeric(df["pageVisitsCountHistory"], downcast="integer")
    df["scrollPercentageHistory"] = pd.to_numeric(df["scrollPercentageHistory"], downcast="float")
    df["minutesSinceLastVisit"] = pd.to_numeric(df["minutesSinceLastVisit"], downcast="float")
    df["timestampHistoryWeekday"] = df["timestampHistoryWeekday"].astype("int16")
    df["timestampHistoryHour"] = df["timestampHistoryHour"].astype("int16")
    return df
