"""
Configurações gerais para o pipeline de feature engineering.
"""

import pandas as pd

# --------------------------------------------------
#  CONFIGURAÇÕES GERAIS
# --------------------------------------------------

FLAG_REMOTE = False  # Define se os dados serão processados localmente ou remotamente
COLD_START_THRESHOLD = 5  # Limite para classificar usuários como "cold start"
SAMPLE_RATE = 0.10  # Fração de amostragem dos dados

# Data de referência
DT_TODAY = pd.Timestamp.today().date()
TODAY = DT_TODAY.strftime("%Y-%m-%d")


# --------------------------------------------------
#  CAMINHOS PARA ARMAZENAMENTO DE DADOS
# --------------------------------------------------

# Diretórios de armazenamento de dados
LOCAL_DATA_PATH = "C:/Users/gufer/OneDrive/Documentos/FIAP/Fase_05/ML_Engineer_Datathon/data/"
REMOTE_DATA_PATH = "s3://..."  # Caminho remoto (S3, GCS, etc.)

# --------------------------------------------------
#  CONFIGURAÇÕES DE ARQUIVOS DE NOTÍCIAS
# --------------------------------------------------

NEWS_TEMP_PATH = "data/challenge-webmedia-e-globo-2023/itens/itens/itens-parte{}.csv"
NEWS_N_CSV_FILES = 3  # Quantidade de arquivos CSV de notícias a serem processados

# --------------------------------------------------
#  CONFIGURAÇÕES DE ARQUIVOS DE USUÁRIOS
# --------------------------------------------------

USERS_TEMP_PATH = "data/challenge-webmedia-e-globo-2023/files/treino/treino_parte{}.csv"
USERS_N_CSV_FILES = 6  # Quantidade de arquivos CSV de usuários a serem processados
