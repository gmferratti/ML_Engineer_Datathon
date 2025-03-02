import pandas as pd
import pytest
import numpy as np

def _compute_time_gap(df_mix: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a diferença de tempo entre o histórico do usuário e a publicação da notícia.
    """
    gap = df_mix['timestampHistoryDatetime'] - df_mix['issuedDatetime']
    df_mix['timeGapDays'] = gap.dt.days
    df_mix['timeGapHours'] = gap / pd.Timedelta(hours=1)
    df_mix['timeGapMinutes'] = gap / pd.Timedelta(minutes=1)
    df_mix['timeGapLessThanOneDay'] = df_mix['timeGapHours'] <= 24
    return df_mix

# Fixture para criar um DataFrame de teste
@pytest.fixture
def sample_df_mix():
    data = {
        'timestampHistoryDatetime': [
            pd.to_datetime('2024-03-10 12:00:00'),
            pd.to_datetime('2024-03-15 18:30:00'),
            pd.to_datetime('2024-03-20 08:00:00'),
            pd.to_datetime('2024-03-10 12:00:00'), # Same timestamp
            pd.NaT,  #Caso Borda: Timestamp faltando.
            pd.to_datetime('2024-03-16 18:30:00'), # Timestamp posterior
        ],
        'issuedDatetime': [
            pd.to_datetime('2024-03-10 10:00:00'),
            pd.to_datetime('2024-03-15 19:00:00'),
            pd.to_datetime('2024-03-20 07:00:00'),
            pd.to_datetime('2024-03-10 12:00:00'), # Same timestamp
            pd.to_datetime('2024-03-16 18:30:00'), #Valor válido, mas HistoryDatetime = NaT
            pd.NaT,  #Caso Borda: Timestamp faltando
        ]
    }
    return pd.DataFrame(data)

# Teste para _compute_time_gap
def test_compute_time_gap(sample_df_mix):
    df_result = _compute_time_gap(sample_df_mix.copy())

    # Verifica se as colunas foram criadas
    assert 'timeGapDays' in df_result.columns
    assert 'timeGapHours' in df_result.columns
    assert 'timeGapMinutes' in df_result.columns
    assert 'timeGapLessThanOneDay' in df_result.columns

     # Verifica os tipos das colunas
    assert df_result['timeGapDays'].dtype == 'float64'
    assert df_result['timeGapHours'].dtype == 'float64'
    assert df_result['timeGapMinutes'].dtype == 'float64'
    assert df_result['timeGapLessThanOneDay'].dtype == 'bool'

    #Verifica os valores (casos específicos, incluindo timedelta)
    assert df_result.loc[0, 'timeGapDays'] == 0
    assert df_result.loc[0, 'timeGapHours'] == 2.0
    assert df_result.loc[0, 'timeGapMinutes'] == 120.0
    assert df_result.loc[0, 'timeGapLessThanOneDay'] == True

    assert df_result.loc[1, 'timeGapDays'] == -1  #Diferença negativa.
    assert df_result.loc[1, 'timeGapHours'] == -0.5
    assert df_result.loc[1, 'timeGapMinutes'] == -30.0
    assert df_result.loc[1, 'timeGapLessThanOneDay'] == True

    assert df_result.loc[2, 'timeGapDays'] == 0
    assert df_result.loc[2, 'timeGapHours'] == 1.0
    assert df_result.loc[2, 'timeGapMinutes'] == 60.0
    assert df_result.loc[2, 'timeGapLessThanOneDay'] == True

    assert df_result.loc[3, 'timeGapDays'] == 0
    assert df_result.loc[3, 'timeGapHours'] == 0.0
    assert df_result.loc[3, 'timeGapMinutes'] == 0.0
    assert df_result.loc[3, 'timeGapLessThanOneDay'] == True

    #Verifica casos com NaT (operações com NaT resultam em NaT)
    assert pd.isna(df_result.loc[4, 'timeGapDays'])
    assert pd.isna(df_result.loc[4, 'timeGapHours'])
    assert pd.isna(df_result.loc[4, 'timeGapMinutes'])
    assert df_result.loc[4, 'timeGapLessThanOneDay'] == False #Importante verificar.

    assert pd.isna(df_result.loc[5, 'timeGapDays'])
    assert pd.isna(df_result.loc[5, 'timeGapHours'])
    assert pd.isna(df_result.loc[5, 'timeGapMinutes'])
    assert df_result.loc[5, 'timeGapLessThanOneDay'] == False #Importante verificar.

#Teste DataFrame vazio:
def test_compute_time_gap_empty():
    df_empty = pd.DataFrame(columns=['timestampHistoryDatetime', 'issuedDatetime'])
    # Especifica os tipos das colunas
    df_empty['timestampHistoryDatetime'] = pd.to_datetime(df_empty['timestampHistoryDatetime'])
    df_empty['issuedDatetime'] = pd.to_datetime(df_empty['issuedDatetime'])
    df_result = _compute_time_gap(df_empty)
    assert df_result.empty
    assert 'timeGapDays' in df_result.columns
    assert 'timeGapHours' in df_result.columns
    assert 'timeGapMinutes' in df_result.columns
    assert 'timeGapLessThanOneDay' in df_result.columns