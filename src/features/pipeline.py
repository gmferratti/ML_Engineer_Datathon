"""Wrapper function to feature engineering pipeline."""
import pandas as pd
from pp_news import pre_process_news
from pp_users import pre_process_users

def pre_process_data() -> None:
    df_news = pre_process_news()
    print(df_news.head())
    df_users = pre_process_users()
    print(df_users.head())
    print("Done!")
    
pre_process_data()