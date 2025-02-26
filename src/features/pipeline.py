"""Wrapper function to feature engineering pipeline."""
import os
from features.pp_news import preprocess_news
from features.pp_users import preprocess_users
from features.pp_target import preprocess_target
from features.pp_mix import preprocess_mix_feats, generate_suggested_feats
from features.feat_selection import feature_selection
from feat_settings import (
    FLAG_REMOTE,
    LOCAL_DATA_PATH, 
    REMOTE_DATA_PATH
)
from config import logger

def pre_process_data() -> None:
    """
    Executa o pipeline de feature engineering:
    1. Pré-processa as informações de news e users.
    2. Gera as mix feats e salva os respectivos arquivos parquet.
    3. Monta as suggested features a partir das dimensões.
    4. Valida a seleção das features para gerar um conjunto final.
    5. Salva o DataFrame final em parquet.
    """
    # Define o caminho dos dados com base na flag de execução remota ou local
    if FLAG_REMOTE:
        DATA_PATH = REMOTE_DATA_PATH
        logger.info("Remote storage chosen!")
    else:
        DATA_PATH = LOCAL_DATA_PATH  
        logger.info("Local storage chosen!")

    logger.info(f"Info will be saved at {DATA_PATH}")    
    
    # Pré-processamento das informações de news
    logger.info("Pre-processing news info...")
    df_news = preprocess_news()
    news_path = os.path.join(DATA_PATH, "features", "news_feats.parquet")
    logger.info(f"Saving pre-processed news parquet at {news_path}...")
    df_news.to_parquet(news_path)
    
    # Pré-processamento das informações de users
    logger.info("Pre-processing users info...")
    df_users = preprocess_users()
    users_path = os.path.join(DATA_PATH, "features", "users_feats.parquet")
    logger.info(f"Saving pre-processed users parquet at {users_path}...")
    df_users.to_parquet(users_path)
    
    # Criando target a partir do df_users
    logger.info("Pre-processing target values...")
    df_target = preprocess_target(df_users)
    target_path = os.path.join(DATA_PATH, "features", "target.parquet")
    logger.info(f"Saving pre-processed target parquet at {target_path}...")
    df_target.to_parquet(target_path)
    
    # Geração das mix feats a partir dos dataframes de news e users
    logger.info("Generating mix feats...")
    # Supondo que preprocess_mix_feats retorne: mix_df, gap_df, state_df, region_df, tm_df, ts_df
    mix_df, gap_df, state_df, region_df, tm_df, ts_df = preprocess_mix_feats(df_news, df_users)
    
    # Salvando os dataframes intermediários de mix feats
    logger.info("Saving some mix feats parquet files...")
    dfs_to_save = {
        "mix_df": mix_df, 
        "gap_feats": gap_df,
        "state_feats": state_df,
        "region_feats": region_df,
        "theme_main_feats": tm_df,
        "theme_sub_feats": ts_df,
    }
    for file_name, df in dfs_to_save.items():
        file_path = os.path.join(DATA_PATH, "features", f"{file_name}.parquet")
        logger.info(f"Saving {file_name} parquet at {file_path}...")
        df.to_parquet(file_path)
    
    # Monta as suggested features a partir dos dataframes das diferentes dimensões
    logger.info("Assembling suggested features df...")
    suggested_feats = generate_suggested_feats(mix_df, gap_df, state_df, region_df, tm_df, ts_df)
    suggested_feats_path = os.path.join(DATA_PATH, "features", "suggested_feats.parquet")
    suggested_feats.to_parquet(suggested_feats_path)
    logger.info(f"Suggested features DF saved at {suggested_feats_path}")

    # Validação da seleção das features para compor as features finais
    logger.info("Validating feature selection...")
    # final_feats = validate_features(suggested_feats)
    
    # Salva o DataFrame final com as features selecionadas
    # logger.info("Assembling final features DF...")
    # final_feats_path = os.path.join(DATA_PATH, "features", "final_feats.parquet")
    # final_feats.to_parquet(final_feats_path)
    # logger.info(f"Final features DF saved at {final_feats_path}")
    
    logger.info("Pre-processing complete!")

pre_process_data()
