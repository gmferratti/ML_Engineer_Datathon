"""Preprocessing for users features."""

import pandas as pd

from constants import (
    COLS_TO_EXPLODE,
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
    - Concatena CSVs.
    - Explode colunas de histórico.
    - Converte colunas para tipos apropriados.
    - Processa informações temporais.
    - Cria variáveis derivadas (ex: minutos desde o último acesso, flag de cold start).
    - Realiza downcasting das colunas.
    """
    # Concatena CSVs
    df_users = concatenate_csv_to_df(USERS_TEMP_PATH, USERS_N_CSV_FILES)
    
    # Faz o sampling dos dados
    df_users = df_users.sample(frac=SAMPLE_RATE, random_state=42)
    
    # Processa colunas de histórico (explode e remove espaços)
    df_users = _process_history_columns(df_users)

    # Converte colunas iniciais para tipos apropriados
    df_users = df_users.astype(USERS_DTYPES)

    # Converte timestamp e ordena por usuário e data
    df_users = _process_timestamp(df_users)

    # Cria variáveis temporais derivadas
    df_users = _extract_time_features(df_users)
    
    # Cria indicador de cold start
    df_users["coldStart"] = df_users["historySize"] < COLD_START_THRESHOLD
    
    # Renomeia a coluna de chave secundária
    df_users.rename(columns={"history": "pageId"}, inplace=True)
    
    # Remove colunas desnecessárias
    df_users.drop(columns=["timestampHistory", "timestampHistory_new"], inplace=True)

    # Cria variável de target
    df_users = _create_target(df_users)

    # Realiza o downcasting das colunas numéricas
    df_users = _downcast_columns(df_users)

    return df_users


def _process_history_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Converte colunas de histórico de string para lista, explode e remove espaços."""
    # Transforma colunas de histórico de string para lista
    df[COLS_TO_EXPLODE] = df[COLS_TO_EXPLODE].apply(lambda col: col.str.split(","))

    # Explode o dataframe e remove espaços das strings
    df = df.explode(COLS_TO_EXPLODE)
    df[COLS_TO_EXPLODE] = df[COLS_TO_EXPLODE].apply(lambda col: col.str.strip())

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
    df["timestampHistoryTime"] = df["timestampHistory"].dt.strftime("%H:%M:%S")
    df["timestampHistoryWeekday"] = df["timestampHistory"].dt.dayofweek
    df["timestampHistoryHour"] = df["timestampHistory"].dt.hour
    
    # Avalia FDS
    df["isWeekend"] = df["timestampHistoryWeekday"] >= 5 
    # Classifica os períodos do dia
    df["dayPeriod"] = _classify_day_period(df)
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

def _create_target(df):
    return df