import pandas as pd
import numpy as np

def preprocess_target(df_users: pd.DataFrame) -> pd.DataFrame:
    """Gera a coluna TARGET padronizada de forma menos sensível a outliers (robust scaling)."""
    agg_df = df_users.groupby('userId', as_index=False).agg({
        'numberOfClicksHistory': 'sum',
        'timeOnPageHistory': 'sum',
        'scrollPercentageHistory': 'mean',
        'pageVisitsCountHistory': 'sum',
        'minutesSinceLastVisit': 'mean'
    })
    
    # Cria a coluna TARGET que é um score de engajamento ponderado
    agg_df['TARGET'] = (
        agg_df['numberOfClicksHistory']
        + (agg_df['timeOnPageHistory'] / 500)
        + agg_df['scrollPercentageHistory']
        + (agg_df['pageVisitsCountHistory'] * 2)
        - (agg_df['minutesSinceLastVisit'] / 50)
    )
    
    # Calcula a mediana dos valores de 'TARGET'
    median_val = agg_df['TARGET'].median()

    # Calcula o intervalo interquartil (IQR)
    iqr_val = agg_df['TARGET'].quantile(0.75) - agg_df['TARGET'].quantile(0.25)

    # Se o IQR for zero (todos os valores iguais, por exemplo), apenas subtrai a mediana
    # Caso contrário, padroniza (robust scaling) subtraindo a mediana e dividindo pelo IQR
    if iqr_val == 0:
        agg_df['TARGET'] = agg_df['TARGET'] - median_val
    else:
        agg_df['TARGET'] = (agg_df['TARGET'] - median_val) / iqr_val

    # Mescla os valores padronizados de 'TARGET' de volta ao DataFrame original
    target_df = df_users.merge(agg_df[['userId', 'TARGET']], on='userId', how='left')

    return target_df
