import pandas as pd
import pytest
import datetime
from pandas.testing import assert_frame_equal

# Função _process_datetime (SIMPLIFICADA)
def _process_datetime(df_news: pd.DataFrame, df_users: pd.DataFrame):
    """Converte datas/horas para datetime e cria timestamps completos."""

    df_news['issuedDate'] = pd.to_datetime(df_news['issuedDate'], format='%Y-%m-%d', errors='coerce')
    df_users['timestampHistoryDate'] = pd.to_datetime(df_users['timestampHistoryDate'], format='%Y-%m-%d', errors='coerce')

    df_news['issuedTime'] = pd.to_datetime(df_news['issuedTime'], format='%H:%M:%S', errors='coerce').dt.time
    df_users['timestampHistoryTime'] = pd.to_datetime(df_users['timestampHistoryTime'], format='%H:%M:%S', errors='coerce').dt.time

    df_news['issuedDatetime'] = df_news['issuedDate'] + df_news['issuedTime'].apply(
        lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second) if pd.notnull(t) else pd.Timedelta(0)
    )
    df_users['timestampHistoryDatetime'] = df_users['timestampHistoryDate'] + df_users['timestampHistoryTime'].apply(
        lambda t: pd.Timedelta(hours=t.hour, minutes=t.minute, seconds=t.second) if pd.notnull(t) else pd.Timedelta(0)
    )

    return df_news, df_users



# Teste 1: Caso Básico (sem valores inválidos)
def test_basic_conversion():
    df_news = pd.DataFrame({
        'issuedDate': ['2024-01-01'],
        'issuedTime': ['10:00:00']
    })
    df_users = pd.DataFrame({
        'timestampHistoryDate': ['2024-01-02'],
        'timestampHistoryTime': ['12:30:00']
    })

    df_news_result, df_users_result = _process_datetime(df_news.copy(), df_users.copy())

    assert df_news_result['issuedDate'].dtype == 'datetime64[ns]'
    assert df_news_result['issuedTime'].dtype == 'object'
    assert df_news_result['issuedDatetime'].dtype == 'datetime64[ns]'

    assert df_users_result['timestampHistoryDate'].dtype == 'datetime64[ns]'
    assert df_users_result['timestampHistoryTime'].dtype == 'object'
    assert df_users_result['timestampHistoryDatetime'].dtype == 'datetime64[ns]'

    assert df_news_result.loc[0, 'issuedDate'] == pd.to_datetime('2024-01-01')
    assert df_news_result.loc[0, 'issuedTime'] == datetime.time(10, 0, 0)
    assert df_news_result.loc[0, 'issuedDatetime'] == pd.to_datetime('2024-01-01 10:00:00')

    assert df_users_result.loc[0, 'timestampHistoryDate'] == pd.to_datetime('2024-01-02')
    assert df_users_result.loc[0, 'timestampHistoryTime'] == datetime.time(12, 30, 0)
    assert df_users_result.loc[0, 'timestampHistoryDatetime'] == pd.to_datetime('2024-01-02 12:30:00')



# Teste 2: Valores Nulos (None)
def test_null_values():
    df_news = pd.DataFrame({
        'issuedDate': [None],
        'issuedTime': [None]
    })
    df_users = pd.DataFrame({
        'timestampHistoryDate': [None],
        'timestampHistoryTime': [None]
    })

    df_news_result, df_users_result = _process_datetime(df_news.copy(), df_users.copy())

    assert pd.isna(df_news_result['issuedDate'][0])
    assert pd.isna(df_news_result['issuedTime'][0])  # CORRIGIDO
    assert pd.isna(df_news_result['issuedDatetime'][0])

    assert pd.isna(df_users_result['timestampHistoryDate'][0])
    assert pd.isna(df_users_result['timestampHistoryTime'][0])  # CORRIGIDO
    assert pd.isna(df_users_result['timestampHistoryDatetime'][0])


# Teste 3: Mistura de Válidos e Nulos
def test_mixed_values():
    df_news = pd.DataFrame({
        'issuedDate': ['2024-01-01', None, '2024-01-03'],
        'issuedTime': ['10:00:00', '15:45:00', None]
    })
    df_users = pd.DataFrame({
        'timestampHistoryDate': ['2024-01-02', '2024-01-04', None],
        'timestampHistoryTime': [None, '12:30:00', '09:15:00']
    })

    df_news_result, df_users_result = _process_datetime(df_news.copy(), df_users.copy())

    # News
    assert df_news_result.loc[0, 'issuedDate'] == pd.to_datetime('2024-01-01')
    assert df_news_result.loc[0, 'issuedTime'] == datetime.time(10, 0, 0)
    assert df_news_result.loc[0, 'issuedDatetime'] == pd.to_datetime('2024-01-01 10:00:00')

    assert pd.isna(df_news_result.loc[1, 'issuedDate'])
    assert df_news_result.loc[1, 'issuedTime'] == datetime.time(15, 45, 0)
    assert pd.isna(df_news_result.loc[1, 'issuedDatetime'])

    assert df_news_result.loc[2, 'issuedDate'] == pd.to_datetime('2024-01-03')
    assert pd.isna(df_news_result.loc[2, 'issuedTime'])  # CORRIGIDO
    assert df_news_result.loc[2, 'issuedDatetime'] == pd.to_datetime('2024-01-03 00:00:00')


    # Users
    assert df_users_result.loc[0, 'timestampHistoryDate'] == pd.to_datetime('2024-01-02')
    assert pd.isna(df_users_result.loc[0, 'timestampHistoryTime'])  # CORRIGIDO
    assert df_users_result.loc[0, 'timestampHistoryDatetime'] == pd.to_datetime('2024-01-02 00:00:00')

    assert df_users_result.loc[1, 'timestampHistoryDate'] == pd.to_datetime('2024-01-04')
    assert df_users_result.loc[1, 'timestampHistoryTime'] == datetime.time(12, 30, 0)
    assert df_users_result.loc[1, 'timestampHistoryDatetime'] == pd.to_datetime('2024-01-04 12:30:00')

    assert pd.isna(df_users_result.loc[2, 'timestampHistoryDate'])
    assert df_users_result.loc[2, 'timestampHistoryTime'] == datetime.time(9, 15, 0)
    assert pd.isna(df_users_result.loc[2, 'timestampHistoryDatetime'])

# Teste 4: DataFrames Vazios
def test_empty_dataframes():
    df_news_empty = pd.DataFrame(columns=["issuedDate", "issuedTime"])
    df_users_empty = pd.DataFrame(columns=["timestampHistoryDate", "timestampHistoryTime"])
    df_news_result, df_users_result = _process_datetime(df_news_empty, df_users_empty)

    assert df_news_result.empty
    assert df_users_result.empty
    assert "issuedDatetime" in df_news_result.columns
    assert "timestampHistoryDatetime" in df_users_result.columns