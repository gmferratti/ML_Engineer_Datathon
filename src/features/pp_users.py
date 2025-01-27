"""Preprocessing for users features."""
import pandas as pd
from utils import concatenate_csv_to_df
from constants import (
    users_template_path, 
    users_num_csv_files,
    users_cols_to_explode,
    users_dtypes
)

def pre_process_users() -> pd.DataFrame:
    """
    Pré-processamento dos dados de usuários.
    """
    # Concatena CSVs
    df_users = concatenate_csv_to_df(
        users_template_path, 
        users_num_csv_files)
    
    # Transforma colunas de historico em listas
    def split_column(series):
        return series.str.split(',')

    for col in users_cols_to_explode:
        df_users[col] = split_column(df_users[col])
        
    # Explode o dataframe
    df_users = df_users.explode(users_cols_to_explode)

    # Tira os espaços das strings
    for col in users_cols_to_explode:
        df_users[col] = df_users[col].str.strip()
        
    # Converte colunas para o tipo correto
    df_users = df_users.convert_dtypes()
    
    return df_users
    
def handle_cold_start():
    return None
    
    