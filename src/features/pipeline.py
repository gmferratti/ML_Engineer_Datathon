"""Wrapper function to feature engineering pipeline."""
import os
from features.pp_news import preprocess_news
from features.pp_users import preprocess_users
from features.pp_mix import preprocess_mix_feats
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
    
    logger.info("Generating mix feats...")
    (gap_df, state_df, region_df, tm_df, ts_df) = preprocess_mix_feats(df_news, df_users)
    
    logger.info("Saving mix feats parquet files...")
    
    dfs_to_save = {
        "gap_feats": gap_df,
        "state_feats": state_df,
        "region_feats": region_df,
        "theme_main_feats": tm_df,
        "theme_sub_feats": ts_df,
    }

    for file_name, df in dfs_to_save.items():
        file_path = os.path.join(f"{DATA_PATH}/features/{file_name}.parquet")
        df.to_parquet(file_path)
    
    logger.info("Validating feature selection...")
    
    
    logger.info("Generating final features DF...")
    
    
    logger.info("Pre-processing complete!")

pre_process_data()
