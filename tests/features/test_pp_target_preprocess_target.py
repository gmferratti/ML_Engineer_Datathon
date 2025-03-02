import pytest
import pandas as pd
import numpy as np
import sys
import os

# Adicione o caminho do projeto ao sys.path
#sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../../..')))

from features.constants import  TARGET_FINAL_COLS
from features.pp_target import preprocess_target  # Ajuste 'features.pp_target' para o caminho correto do módulo
from src.config import SCALING_RANGE

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
