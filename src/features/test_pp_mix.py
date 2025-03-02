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

import pandas as pd
import pytest
import numpy as np  # Importante para usar np.nan
from pandas.testing import assert_frame_equal

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
        count_col = f'count{col_title}User' # 'LocalState' -> 'countLocalStateUser'
        # Usa o .transform('count') que lida corretamente com NaNs.
        df_mix[count_col] = df_mix.groupby(['userId', col])['pageId'].transform('count')

    #Contabiliza o total de pageId por usuario
    df_mix['totalUserNews'] = df_mix.groupby('userId')['pageId'].transform('count')

    # Preenche NaNs com 0 *ANTES* de calcular as proporções.  Isso evita erros.
    for col in category_columns:
        col_title = col[0].upper() + col[1:]
        count_col = f'count{col_title}User'
        df_mix[count_col] = df_mix[count_col].fillna(0).astype(int)  # <---  IMPORTANTE!

    # Cria colunas de porcentagem relativa ao total de notícias consumidas.
    for col in category_columns:
        col_title = col[0].upper() + col[1:]
        count_col = f'count{col_title}User'
        rel_col = f'rel{col_title}'
        # O fillna garante que, se count for NaN (devido a todos os valores serem NaN
        # no grupo), o resultado da divisão será NaN, e não um erro.

        df_mix[rel_col] = df_mix[count_col] / df_mix['totalUserNews']
        df_mix[rel_col] = df_mix[rel_col].fillna(0) #Adicionado

    return df_mix




# Fixture para criar um DataFrame de teste
@pytest.fixture
def sample_df_mix():
    data = {
        'userId': [1, 1, 1, 2, 2, 3, 3, 3, 3],
        'pageId': [101, 102, 103, 104, 105, 106, 107, 108, 109],
        'localState': ['SP', 'SP', 'RJ', 'MG', 'MG', 'SP', 'RJ', 'RS', 'RS'],
        'localRegion': ['A', 'A', 'B', 'C', 'C', 'A', 'B', 'D', 'D'],
        'themeMain': ['X', 'X', 'Y', 'Z', 'Z', 'X', 'Y', 'W', 'W'],
        'themeSub': ['P', 'Q', 'R', 'S', 'T', 'P', 'U', 'V', 'V']
    }
    return pd.DataFrame(data)

# Teste principal
def test_compute_category_counts(sample_df_mix):
    df_result = _compute_category_counts(sample_df_mix.copy())

    # 1. Verifica se as colunas foram criadas
    expected_count_cols = ['countLocalStateUser', 'countLocalRegionUser', 'countThemeMainUser', 'countThemeSubUser']
    expected_rel_cols = ['relLocalState', 'relLocalRegion', 'relThemeMain', 'relThemeSub']
    assert all(col in df_result.columns for col in expected_count_cols + expected_rel_cols + ['totalUserNews'])


    # 2. Verifica os tipos das colunas
    for col in expected_count_cols + ['totalUserNews']:
        assert df_result[col].dtype == 'int64'
    for col in expected_rel_cols:
        assert df_result[col].dtype == 'float64'


    # 3. Verifica os valores (casos calculados manualmente)
    # Usuário 1
    assert df_result.loc[0, 'totalUserNews'] == 3
    assert df_result.loc[0, 'countLocalStateUser'] == 2  # SP
    assert df_result.loc[0, 'countLocalRegionUser'] == 2 # A
    assert df_result.loc[0, 'countThemeMainUser'] == 2   # X
    assert df_result.loc[0, 'countThemeSubUser'] == 1    # P
    assert df_result.loc[0, 'relLocalState'] == 2/3
    assert df_result.loc[0, 'relLocalRegion'] == 2/3
    assert df_result.loc[0, 'relThemeMain'] == 2/3
    assert df_result.loc[0, 'relThemeSub'] == 1/3

    # Usuário 2
    assert df_result.loc[3, 'totalUserNews'] == 2
    assert df_result.loc[3, 'countLocalStateUser'] == 2 # MG
    assert df_result.loc[3, 'countLocalRegionUser'] == 2# C
    assert df_result.loc[3, 'countThemeMainUser'] == 2  # Z
    assert df_result.loc[3, 'countThemeSubUser'] == 1   # S
    assert df_result.loc[3, 'relLocalState'] == 1.0
    assert df_result.loc[3, 'relLocalRegion'] == 1.0
    assert df_result.loc[3, 'relThemeMain'] == 1.0
    assert df_result.loc[3, 'relThemeSub'] == 0.5

    # Usuário 3
    assert df_result.loc[5, 'totalUserNews'] == 4
    assert df_result.loc[5, 'countLocalStateUser'] == 1 # SP
    assert df_result.loc[5, 'countLocalRegionUser'] == 1# A
    assert df_result.loc[5, 'countThemeMainUser'] == 1  # X
    assert df_result.loc[5, 'countThemeSubUser'] == 1   # P
    assert df_result.loc[5, 'relLocalState'] == 1/4
    assert df_result.loc[5, 'relLocalRegion'] == 1/4
    assert df_result.loc[5, 'relThemeMain'] == 1/4
    assert df_result.loc[5, 'relThemeSub'] == 1/4

    # 4. Teste com category_columns customizado
    df_result_custom = _compute_category_counts(sample_df_mix.copy(), category_columns=['localState', 'themeMain'])
    assert 'countLocalStateUser' in df_result_custom.columns
    assert 'countThemeMainUser' in df_result_custom.columns
    assert 'relLocalState' in df_result_custom.columns
    assert 'relThemeMain' in df_result_custom.columns
     # Verifica se as colunas das categorias não especificadas NÃO foram criadas
    assert 'countLocalRegionUser' not in df_result_custom.columns
    assert 'countThemeSubUser' not in df_result_custom.columns
    assert 'relLocalRegion' not in df_result_custom.columns
    assert 'relThemeSub' not in df_result_custom.columns


# Teste com DataFrame vazio
def test_compute_category_counts_empty():
    df_empty = pd.DataFrame(columns=['userId', 'pageId', 'localState', 'localRegion', 'themeMain', 'themeSub'])
    df_result = _compute_category_counts(df_empty)
    assert df_result.empty
    # Verifica se as colunas são criadas mesmo com o DF vazio.
    expected_count_cols = ['countLocalStateUser', 'countLocalRegionUser', 'countThemeMainUser', 'countThemeSubUser']
    expected_rel_cols = ['relLocalState', 'relLocalRegion', 'relThemeMain', 'relThemeSub']
    assert all(col in df_result.columns for col in expected_count_cols + expected_rel_cols + ['totalUserNews'])


# Teste com valores faltantes (NaN) nas categorias
def test_compute_category_counts_with_nan():
    data = {
        'userId': [1, 1, 1, 2, 2],
        'pageId': [101, 102, 103, 104, 105],
        'localState': ['SP', 'SP', None, 'MG', 'MG'],  # None em localState
        'localRegion': ['A', 'A', 'B', None, None],  # None em localRegion
        'themeMain': [None, None, 'Y', 'Z', 'Z'],    # None em themeMain
        'themeSub': ['P', None, 'R', 'S', 'T']        # None em themeSub
    }
    df_mix = pd.DataFrame(data)
    df_result = _compute_category_counts(df_mix.copy())

    # Usuário 1, localState = None (linha 2)
    assert df_result.loc[2, 'countLocalStateUser'] == 0  # Foi preenchido com 0
    assert df_result['totalUserNews'][2] == 3 #Total de noticias do user 1
    assert df_result.loc[2, 'relLocalState'] ==  0.0 #Divisão por 0

    # localRegion B, user 1
    assert df_result.loc[2, 'countLocalRegionUser'] == 1
    assert df_result.loc[2, 'relLocalRegion'] == 1/3

#6---------
import pytest
import pandas as pd
from features.pp_mix import _split_dataframes  # Ajuste 'features.pp_mix' para o caminho correto

from features.constants import (
    GAP_COLS, STATE_COLS, REGION_COLS,
    THEME_MAIN_COLS, THEME_SUB_COLS
)

@pytest.fixture
def sample_df_mix():
    data = {
        'userId': [1, 1, 2, 2],
        'pageId': [1, 2, 3, 4],
        'timeGapDays': [1, -1, 2, 0],
        'timeGapHours': [24, 36, 48, 12],
        'timeGapMinutes': [1440, 2160, 2880, 720],
        'timeGapLessThanOneDay': [False, False, False, True],
        'countLocalStateUser': [1, 0, 2, 1],
        'countLocalRegionUser': [1, 2, 0, 2],
        'countThemeMainUser': [0, 1, 2, 1],
        'countThemeSubUser': [1, 0, 1, 2],
        'localState': ['SP', 'RJ', 'MG', 'SP'],
        'localRegion': ['Sul', 'Norte', 'Leste', 'Oeste'],
        'themeMain': ['Economia', 'Política', 'Saúde', 'Tecnologia'],
        'themeSub': ['Impostos', 'Eleição', 'Pandemia', 'Inovação'],
        'relLocalState': [0.5, 0.2, 0.7, 0.3],  # Adicionando coluna faltante
        'relLocalRegion': [0.4, 0.5, 0.6, 0.7],
        'relThemeMain': [0.3, 0.4, 0.5, 0.6],
        'relThemeSub': [0.2, 0.3, 0.4, 0.5],
    }
    return pd.DataFrame(data)

def test_split_dataframes(sample_df_mix):
    gap_df, state_df, region_df, tm_df, ts_df = _split_dataframes(sample_df_mix)

    # Verifica se os DataFrames não estão vazios
    assert not gap_df.empty, "gap_df está vazio"
    assert not state_df.empty, "state_df está vazio"
    assert not region_df.empty, "region_df está vazio"
    assert not tm_df.empty, "tm_df está vazio"
    assert not ts_df.empty, "ts_df está vazio"

    # Verifica se os DataFrames foram filtrados corretamente
    assert all(gap_df['timeGapDays'] >= 0), "Existem valores negativos em timeGapDays no gap_df"
    assert all(state_df['countLocalStateUser'] > 0), "Existem valores não positivos em countLocalStateUser no state_df"
    assert all(region_df['countLocalRegionUser'] > 0), "Existem valores não positivos em countLocalRegionUser no region_df"
    assert all(tm_df['countThemeMainUser'] > 0), "Existem valores não positivos em countThemeMainUser no tm_df"
    assert all(ts_df['countThemeSubUser'] > 0), "Existem valores não positivos em countThemeSubUser no ts_df"

    # Verifica se as colunas esperadas estão presentes
    expected_gap_cols = GAP_COLS
    expected_state_cols = STATE_COLS
    expected_region_cols = REGION_COLS
    expected_tm_cols = THEME_MAIN_COLS
    expected_ts_cols = THEME_SUB_COLS

    for col in expected_gap_cols:
        assert col in gap_df.columns, f"Coluna {col} não encontrada em gap_df"
    
    for col in expected_state_cols:
        assert col in state_df.columns, f"Coluna {col} não encontrada em state_df"
    
    for col in expected_region_cols:
        assert col in region_df.columns, f"Coluna {col} não encontrada em region_df"

    for col in expected_tm_cols:
        assert col in tm_df.columns, f"Coluna {col} não encontrada em tm_df"

    for col in expected_ts_cols:
        assert col in ts_df.columns, f"Coluna {col} não encontrada em ts_df"

if __name__ == "__main__":
    pytest.main()


    