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



import pandas as pd
import pytest
from pandas.testing import assert_frame_equal
from features.pp_mix import preprocess_mix_feats, _process_datetime, _compute_time_gap, _compute_category_counts, _split_dataframes
from features.constants import MIX_FEATS_COLS, GAP_COLS, STATE_COLS, REGION_COLS, THEME_MAIN_COLS, THEME_SUB_COLS, FINAL_MIX_FEAT_COLS
import copy
import datetime
from pandas import DataFrame

# Fixture para os dados de entrada (originais)
@pytest.fixture
def original_dataframes():
    df_news_original = pd.DataFrame({
        "pageId": [1, 2, 3],
        "issuedDate": ["2024-03-01", "2024-03-05", "2024-03-10"],
        "issuedTime": ["10:00:00", "14:00:00", "18:00:00"],
        "localState": ["SP", "RJ", "MG"],
        "localRegion": ["Sudeste", "Sudeste", "Sudeste"],
        "themeMain": ["Esportes", "Política", "Economia"],
        "themeSub": ["Futebol", "Eleições", "Inflação"]
    })
    df_users_original = pd.DataFrame({
        "userId": [1, 2, 3, 1],
        "pageId": [1, 2, 3, 1],  # Note o pageId 1 duplicado para o user 1
        "timestampHistoryDate": ["2024-03-02", "2024-03-06", "2024-03-11", "2024-03-03"],
        "timestampHistoryTime": ["11:00:00", "15:00:00", "19:00:00", "12:00:00"],
        "userType": ["Free", "Premium", "Free", "Premium"],
        "coldStart": [True, False, True, False],
        "historySize": [1, 12, 5, 10],
        "isWeekend": [False, True, False, True],
        "dayPeriod": ["Manhã", "Tarde", "Noite", "Tarde"]
    })
    return copy.deepcopy(df_news_original), copy.deepcopy(df_users_original)


def test_preprocess_mix_feats_df_mix(original_dataframes: tuple[DataFrame, DataFrame]):
    """Testa preprocess_mix_feats, verificando df_mix."""
    df_news_original, df_users_original = original_dataframes
    df_mix_result, _, _, _, _, _ = preprocess_mix_feats(df_news_original, df_users_original)


    # Calcula o df_mix esperado *chamando as funções auxiliares*
    df_news, df_users = _process_datetime(df_news_original.copy(), df_users_original.copy())
    df_mix_expected = pd.merge(df_users, df_news, on='pageId', how='inner')[MIX_FEATS_COLS]
    df_mix_expected = _compute_time_gap(df_mix_expected)       # <--- Adicionado!
    df_mix_expected = _compute_category_counts(df_mix_expected)  # <--- Adicionado!
    assert_frame_equal(df_mix_result, df_mix_expected, check_dtype=False)



def test_preprocess_mix_feats_gap_df(original_dataframes: tuple[DataFrame, DataFrame]):
    """Testa preprocess_mix_feats, verificando gap_df."""
    df_news_original, df_users_original = original_dataframes
    _, gap_df_result, _, _, _, _ = preprocess_mix_feats(df_news_original, df_users_original)

     # Calcula o gap_df esperado *chamando as funções auxiliares*
    df_news, df_users = _process_datetime(df_news_original.copy(), df_users_original.copy())
    df_mix = pd.merge(df_users, df_news, on='pageId', how='inner')[MIX_FEATS_COLS]
    df_mix_enriched = _compute_time_gap(df_mix)
    df_mix_enriched = _compute_category_counts(df_mix_enriched)
    gap_df_expected, _, _, _, _ = _split_dataframes(df_mix_enriched)

    assert_frame_equal(gap_df_result, gap_df_expected, check_dtype=False)

# ... Os outros testes (state_df, region_df, etc.) seguem a mesma lógica ...
# (Repita a mesma estrutura: função de teste separada, calcular o DataFrame esperado
#  chamando *todas* as funções auxiliares necessárias, e usar assert_frame_equal)

def test_preprocess_mix_feats_state_df(original_dataframes: tuple[DataFrame, DataFrame]):
    """Testa preprocess_mix_feats, verificando state_df."""
    df_news_original, df_users_original = original_dataframes
    _, _, state_df_result, _, _, _ = preprocess_mix_feats(df_news_original, df_users_original)

    df_news, df_users = _process_datetime(df_news_original.copy(), df_users_original.copy())
    df_mix = pd.merge(df_users, df_news, on='pageId', how='inner')[MIX_FEATS_COLS]
    df_mix_enriched = _compute_time_gap(df_mix)
    df_mix_enriched = _compute_category_counts(df_mix_enriched)
    _, state_df_expected, _, _, _ = _split_dataframes(df_mix_enriched)

    assert_frame_equal(state_df_result, state_df_expected, check_dtype=False)

def test_preprocess_mix_feats_region_df(original_dataframes: tuple[DataFrame, DataFrame]):
    """Testa preprocess_mix_feats, verificando region_df."""
    df_news_original, df_users_original = original_dataframes
    _, _, _, region_df_result, _, _ = preprocess_mix_feats(df_news_original, df_users_original)

    df_news, df_users = _process_datetime(df_news_original.copy(), df_users_original.copy())
    df_mix = pd.merge(df_users, df_news, on='pageId', how='inner')[MIX_FEATS_COLS]
    df_mix_enriched = _compute_time_gap(df_mix)
    df_mix_enriched = _compute_category_counts(df_mix_enriched)
    _, _, region_df_expected, _, _ = _split_dataframes(df_mix_enriched)

    assert_frame_equal(region_df_result, region_df_expected, check_dtype=False)

def test_preprocess_mix_feats_tm_df(original_dataframes: tuple[DataFrame, DataFrame]):
    """Testa preprocess_mix_feats, verificando tm_df."""
    df_news_original, df_users_original = original_dataframes
    _, _, _, _, tm_df_result, _ = preprocess_mix_feats(df_news_original, df_users_original)

    df_news, df_users = _process_datetime(df_news_original.copy(), df_users_original.copy())
    df_mix = pd.merge(df_users, df_news, on='pageId', how='inner')[MIX_FEATS_COLS]
    df_mix_enriched = _compute_time_gap(df_mix)
    df_mix_enriched = _compute_category_counts(df_mix_enriched)
    _, _, _, tm_df_expected, _ = _split_dataframes(df_mix_enriched)

    assert_frame_equal(tm_df_result, tm_df_expected, check_dtype=False)

def test_preprocess_mix_feats_ts_df(original_dataframes: tuple[DataFrame, DataFrame]):
    """Testa preprocess_mix_feats, verificando ts_df."""
    df_news_original, df_users_original = original_dataframes
    _, _, _, _, _, ts_df_result = preprocess_mix_feats(df_news_original, df_users_original)

    df_news, df_users = _process_datetime(df_news_original.copy(), df_users_original.copy())
    df_mix = pd.merge(df_users, df_news, on='pageId', how='inner')[MIX_FEATS_COLS]
    df_mix_enriched = _compute_time_gap(df_mix)
    df_mix_enriched = _compute_category_counts(df_mix_enriched)
    _, _, _, _, ts_df_expected = _split_dataframes(df_mix_enriched)

    assert_frame_equal(ts_df_result, ts_df_expected, check_dtype=False)