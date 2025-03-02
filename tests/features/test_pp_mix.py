#1------------

import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

# Supondo que você tenha importado sua função generate_suggested_feats corretamente
from features.pp_mix import generate_suggested_feats

# Importe as constantes de colunas
from features.constants import FINAL_MIX_FEAT_COLS, STATE_COLS, REGION_COLS, THEME_MAIN_COLS, THEME_SUB_COLS

@pytest.fixture
def df_mix():
    return pd.DataFrame({
        "userId": [1],
        "pageId": [100],
        "userType": ["Free"],
        "isWeekend": [False],
        "dayPeriod": ["Manhã"],
        "issuedDatetime": [pd.to_datetime("2024-03-01 10:00:00")],
        "timestampHistoryDatetime": [pd.to_datetime("2024-03-01 10:30:00")],
        "coldStart": [False],
        "localState": ["SP"],
        "localRegion": ["Sudeste"],
        "themeMain": ["Esportes"],
        "themeSub": ["Futebol"]
    })

@pytest.fixture
def state_df():
    return pd.DataFrame({
        "userId": [1],
        "localState": ["SP"],
        "countLocalStateUser": [5],
        "relLocalState":''
    })

@pytest.fixture
def region_df():
    return pd.DataFrame({
        "userId": [1],
        "localRegion": ["Sudeste"],
        "countLocalRegionUser": [10],
        "relLocalRegion":''
    })

@pytest.fixture
def tm_df():
    return pd.DataFrame({
        "userId": [1],
        "themeMain": ["Esportes"],
        "countThemeMainUser": [8],
        "relThemeMain":''
    })

@pytest.fixture
def ts_df():
    return pd.DataFrame({
        "userId": [1],
        "themeSub": ["Futebol"],
        "countThemeSubUser": [4],
        "relThemeSub":''
    })

def test_generate_suggested_feats_simple(df_mix, state_df, region_df, tm_df, ts_df):
    result = generate_suggested_feats(df_mix, state_df, region_df, tm_df, ts_df)

    # Remova as colunas 'count' do resultado
    result = result.drop(columns=[col for col in result.columns if col.startswith("count")])

    expected_result = pd.DataFrame({
        "userId": [1],
        "pageId": [100],
        "userType": ["Free"],
        "isWeekend": [False],
        "dayPeriod": ["Manhã"],
        "issuedDatetime": [pd.to_datetime("2024-03-01 10:00:00")],
        "timestampHistoryDatetime": [pd.to_datetime("2024-03-01 10:30:00")],
        "coldStart": [False],
        "localState": ["SP"],
        "localRegion": ["Sudeste"],
        "themeMain": ["Esportes"],
        "themeSub": ["Futebol"],
        "relLocalState":'',
        "relLocalRegion":'',
        "relThemeMain":'',
        "relThemeSub":''
    })

    assert_frame_equal(result, expected_result)


#2-------

import pytest
import pandas as pd
import numpy as np
import sys
import os

# Adicione o caminho do projeto ao sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from features.constants import TARGET_INIT_COLS, TARGET_FINAL_COLS, DEFAULT_TARGET_VALUES
from features.pp_target import preprocess_target  # Ajuste 'features.pp_target' para o caminho correto do módulo
from config import SCALING_RANGE

@pytest.fixture
def sample_df_users():
    data = {
        'userId': [1, 2],
        'pageId': [1, 2],
        'numberOfClicksHistory': [5, 3],
        'timeOnPageHistory': [300, 600],
        'scrollPercentageHistory': [50, 75],
        'minutesSinceLastVisit': [30, 45],
        'historySize': [120, 130],
        'coldStart': [0, 1],  # Adicionando a coluna 'coldStart'
    }
    return pd.DataFrame(data)

@pytest.fixture
def sample_gap_df():
    data = {
        'userId': [1, 2],
        'pageId': [1, 2],
        'timeGapDays': [10, 5],
    }
    return pd.DataFrame(data)

def test_preprocess_target(sample_df_users, sample_gap_df):
    target_df = preprocess_target(sample_df_users, sample_gap_df)

    # Verifica se a saída não está vazia
    assert not target_df.empty, "target_df está vazio"

    # Verifica se as colunas esperadas estão presentes
    expected_cols = TARGET_FINAL_COLS
    for col in expected_cols:
        assert col in target_df.columns, f"Coluna {col} não encontrada em target_df"

    # Verifica se os valores TARGET são dentro do intervalo esperado
    assert target_df['TARGET'].between(0, SCALING_RANGE).all(), "Valores de TARGET fora do intervalo esperado"

if __name__ == "__main__":
    pytest.main()

#3------

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

#4---------

import pandas as pd
import pytest
import logging

# Supondo que o código fornecido esteja em um arquivo chamado 'seu_modulo.py'
# E que 'seu_modulo.py' contenha a função _compute_time_gap e um logger
# (para simular, vamos criar um logger simples)
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

@pytest.fixture
def sample_mix_df():
    return pd.DataFrame({
        'userId': ['A', 'B', 'C'],
        'pageId': [1, 2, 3],
        'issuedDatetime': pd.to_datetime(['2023-01-01 10:00:00', '2023-01-02 12:00:00', '2023-01-03 14:00:00']),
        'timestampHistoryDatetime': pd.to_datetime(['2023-01-01 11:00:00', '2023-01-02 11:00:00', '2023-01-04 15:00:00']),
    })

def test_compute_time_gap(sample_mix_df):
    df_with_gap = _compute_time_gap(sample_mix_df.copy())

    # Verificar se as colunas de gap foram adicionadas
    assert 'timeGapDays' in df_with_gap.columns
    assert 'timeGapHours' in df_with_gap.columns
    assert 'timeGapMinutes' in df_with_gap.columns
    assert 'timeGapLessThanOneDay' in df_with_gap.columns

    # Verificar os valores calculados para o primeiro registro
    assert df_with_gap['timeGapDays'].iloc[0] == 0
    assert df_with_gap['timeGapHours'].iloc[0] == 1
    assert df_with_gap['timeGapMinutes'].iloc[0] == 60
    assert df_with_gap['timeGapLessThanOneDay'].iloc[0] == True

    # Verificar os valores calculados para o segundo registro
    assert df_with_gap['timeGapDays'].iloc[1] == -1
    assert df_with_gap['timeGapHours'].iloc[1] == -1
    assert df_with_gap['timeGapMinutes'].iloc[1] == -60
    assert df_with_gap['timeGapLessThanOneDay'].iloc[1] == True

    # Verificar os valores calculados para o terceiro registro
    assert df_with_gap['timeGapDays'].iloc[2] == 1
    assert df_with_gap['timeGapHours'].iloc[2] == 25
    assert df_with_gap['timeGapMinutes'].iloc[2] == 1500
    assert df_with_gap['timeGapLessThanOneDay'].iloc[2] == False

#5--------

def _compute_category_counts(df_mix: pd.DataFrame, category_columns=None) -> pd.DataFrame:
    """
    Conta quantas notícias o usuário consumiu por categoria (ex.: estado, região),
    cria colunas de quantidade e porcentagem relativa.
    """
    if category_columns is None:
        category_columns = ['localState', 'localRegion', 'themeMain', 'themeSub']

    # Conta ocorrências de cada categoria por usuário
    for col in category_columns:
        col_title = col[0].upper() + col[1:]  # 'localState' -> 'LocalState'
        count_col = f'count{col_title}User'  # 'LocalState' -> 'countLocalStateUser'
        # Usa o .transform('count') que lida corretamente com NaNs.
        df_mix[count_col] = df_mix.groupby(['userId', col])['pageId'].transform('count')

    # Contabiliza o total de pageId por usuário
    df_mix['totalUserNews'] = df_mix.groupby('userId')['pageId'].transform('count')

    # Preenche NaNs com 0 *ANTES* de calcular as proporções. Isso evita erros.
    for col in category_columns:
        col_title = col[0].upper() + col[1:]
        count_col = f'count{col_title}User'
        df_mix[count_col] = df_mix[count_col].fillna(0).astype(int)  # <--- IMPORTANTE!

    # Cria colunas de porcentagem relativa ao total de notícias consumidas.
    for col in category_columns:
        col_title = col[0].upper() + col[1:]
        count_col = f'count{col_title}User'
        rel_col = f'rel{col_title}'
        # O fillna garante que, se count for NaN (devido a todos os valores serem NaN
        # no grupo), o resultado da divisão será NaN, e não um erro.
        df_mix[rel_col] = df_mix[count_col] / df_mix['totalUserNews']
        df_mix[rel_col] = df_mix[rel_col].fillna(0)  # Adicionado

    return df_mix

#6---------
def _split_dataframes(df_mix: pd.DataFrame):
    """
    Separa o DataFrame em subconjuntos com base nas colunas de gap, estado, região, tema principal e tema secundário.
    """
    logger.info("🔀 [Mix] Separando subconjuntos...")

    # Filtra os DataFrames com base nas colunas relevantes
    gap_df = df_mix[GAP_COLS].copy()
    state_df = df_mix[STATE_COLS].copy()
    region_df = df_mix[REGION_COLS].copy()
    tm_df = df_mix[THEME_MAIN_COLS].copy()
    ts_df = df_mix[THEME_SUB_COLS].copy()

    logger.info("🔀 [Mix] Subconjuntos separados.")
    return gap_df, state_df, region_df, tm_df, ts_df


    