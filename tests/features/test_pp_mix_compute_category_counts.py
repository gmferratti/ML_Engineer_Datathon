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

    