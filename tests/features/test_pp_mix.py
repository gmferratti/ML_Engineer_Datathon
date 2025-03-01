import pandas as pd
from pandas.testing import assert_frame_equal
import pytest

# Supondo que você tenha importado sua função generate_suggested_feats corretamente
from features.pp_mix import generate_suggested_feats

# Importe as constantes de colunas
from features.constants import (
    FINAL_MIX_FEAT_COLS,
    STATE_COLS,
    REGION_COLS,
    THEME_MAIN_COLS,
    THEME_SUB_COLS,
)


@pytest.fixture
def df_mix():
    return pd.DataFrame(
        {
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
        }
    )


@pytest.fixture
def state_df():
    return pd.DataFrame(
        {"userId": [1], "localState": ["SP"], "countLocalStateUser": [5], "relLocalState": ""}
    )


@pytest.fixture
def region_df():
    return pd.DataFrame(
        {
            "userId": [1],
            "localRegion": ["Sudeste"],
            "countLocalRegionUser": [10],
            "relLocalRegion": "",
        }
    )


@pytest.fixture
def tm_df():
    return pd.DataFrame(
        {"userId": [1], "themeMain": ["Esportes"], "countThemeMainUser": [8], "relThemeMain": ""}
    )


@pytest.fixture
def ts_df():
    return pd.DataFrame(
        {"userId": [1], "themeSub": ["Futebol"], "countThemeSubUser": [4], "relThemeSub": ""}
    )


def test_generate_suggested_feats_simple(df_mix, state_df, region_df, tm_df, ts_df):
    result = generate_suggested_feats(df_mix, state_df, region_df, tm_df, ts_df)

    # Remova as colunas 'count' do resultado
    result = result.drop(columns=[col for col in result.columns if col.startswith("count")])

    expected_result = pd.DataFrame(
        {
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
            "relLocalState": "",
            "relLocalRegion": "",
            "relThemeMain": "",
            "relThemeSub": "",
        }
    )

    assert_frame_equal(result, expected_result)
