"""Preprocessing for news features."""

import pandas as pd
from utils import concatenate_csv_to_df
from constants import (news_template_path, news_num_csv_files)


def pre_process_news() -> pd.DataFrame:
    """
    Pré-processamento dos dados de notícias.
    """   
    
    df_news = concatenate_csv_to_df(news_template_path, news_num_csv_files)
    
    return df_news