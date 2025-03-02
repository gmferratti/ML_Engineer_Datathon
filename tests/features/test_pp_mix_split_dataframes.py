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
