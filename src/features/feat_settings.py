"""
Configurações gerais para o pipeline de feature engineering.
"""
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
