"""General Features Settings File"""
import pandas as pd
from datetime import datetime

# GENERAL SETTINGS

FLAG_REMOTE = False
COLD_START_THRESHOLD = 5
SAMPLE_RATE = 0.10

DT_TODAY = pd.Timestamp.today().date()
TODAY = DT_TODAY.strftime('%Y-%m-%d')

# PATHS

LOCAL_DATA_PATH = "C:/Users/gufer/OneDrive/Documentos/FIAP/Fase_05/ML_Engineer_Datathon/data/processed_data"
REMOTE_DATA_PATH = "s3://..."

NEWS_TEMP_PATH = "data/challenge-webmedia-e-globo-2023/itens/itens/itens-parte{}.csv"
NEWS_N_CSV_FILES = 3

USERS_TEMP_PATH = "data/challenge-webmedia-e-globo-2023/files/treino/treino_parte{}.csv"
USERS_N_CSV_FILES = 6
