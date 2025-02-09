"""Wrapper function to feature engineering pipeline."""

from .pp_news import preprocess_news
from .pp_users import preprocess_users


def pre_process_data() -> None:
    df_news = preprocess_news()
    print(df_news.head())
    df_users = preprocess_users()
    print(df_users.head())
    print("Done!")


pre_process_data()
