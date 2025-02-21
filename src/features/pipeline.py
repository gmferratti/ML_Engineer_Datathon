"""Wrapper function to feature engineering pipeline."""

from features.pp_news import preprocess_news
from features.pp_users import preprocess_users
from features.pp_mix_feats import preprocess_mix_feats
from feat_settings import (
    FLAG_REMOTE,
    LOCAL_DATA_PATH, 
    REMOTE_DATA_PATH
)
from config import logger

def pre_process_data() -> None:
    
    if FLAG_REMOTE:
        DATA_PATH = REMOTE_DATA_PATH
        logger.info("Remote storage chosen!")
    else:
        DATA_PATH = LOCAL_DATA_PATH  
        logger.info("Local storage chosen!")

    logger.info(f"Info will be saved at {DATA_PATH}")    
    
    logger.info("Pre-processing news info...")
    df_news = preprocess_news()
    
    logger.info("Saving pre-proceessed news parquet...")
    df_news.to_parquet(f"{DATA_PATH}/features/news.parquet")
    
    logger.info("Pre-processing users info...")
    df_users = preprocess_users()
    
    logger.info("Saving pre-proceessed users parquet...")
    df_users.to_parquet(f"{DATA_PATH}/features/users.parquet")
    
    logger.info("Generating combined features...")
    df_combined_feats = preprocess_mix_feats(df_news, df_users)
    df_combined_feats.to_parquet(f"{DATA_PATH}/features/combined_feats.parquet")
    
    logger.info("Pre-processing complete!")

pre_process_data()
