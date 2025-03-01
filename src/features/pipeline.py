import os
from config import logger, USE_S3, DATA_PATH, LOCAL_DATA_PATH
from features.pp_mix import preprocess_mix_feats, generate_suggested_feats
from features.pp_news import preprocess_news
from features.pp_target import preprocess_target
from features.pp_users import preprocess_users


def _get_data_path() -> str:
    """
    Retorna o caminho dos dados com base na flag.

    Returns:
        str: Caminho dos dados.
    """
    if USE_S3:
        logger.info("Armazenamento remoto (S3) selecionado.")
        return DATA_PATH
    logger.info("Armazenamento local selecionado.")
    return LOCAL_DATA_PATH


def _save_df_parquet(df, file_path: str) -> None:
    """
    Salva um DataFrame como Parquet.

    Args:
        df: DataFrame a salvar.
        file_path (str): Caminho para salvar.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path)
    rel_path = os.path.relpath(file_path)
    logger.info("Arquivo salvo: %s", rel_path)


def _preprocess_and_save_news(data_path: str, selected_pageIds: list):
    """
    Pré-processa notícias e salva os dados.

    Args:
        data_path (str): Caminho base.
        selected_pageIds (list): IDs das páginas.

    Returns:
        pd.DataFrame: Dados de notícias.
    """
    logger.info("Iniciando pré-processamento das notícias...")
    news_df = preprocess_news(selected_pageIds)
    news_path = os.path.join(data_path, "features", "news_feats.parquet")
    _save_df_parquet(news_df, news_path)
    logger.info("Notícias: %d linhas | %d páginas únicas",
                news_df.shape[0], news_df["pageId"].nunique())
    return news_df


def _preprocess_and_save_users(data_path: str):
    """
    Pré-processa usuários e salva os dados.

    Args:
        data_path (str): Caminho base.

    Returns:
        pd.DataFrame: Dados dos usuários.
    """
    logger.info("Iniciando pré-processamento dos usuários...")
    users_df = preprocess_users()
    users_path = os.path.join(data_path, "features", "users_feats.parquet")
    _save_df_parquet(users_df, users_path)
    logger.info("Usuários: %d linhas | %d páginas | %d usuários",
                users_df.shape[0], users_df["pageId"].nunique(),
                users_df["userId"].nunique())
    return users_df


def _preprocess_and_save_mix_feats(data_path: str, news_df, users_df):
    """
    Gera mix_feats combinando notícias e usuários.

    Args:
        data_path (str): Caminho base.
        news_df: Dados de notícias.
        users_df: Dados dos usuários.

    Returns:
        tuple: (mix_df, gap_df, state_df, region_df, tm_df, ts_df)
    """
    logger.info("Gerando mix_feats...")
    mix_df, gap_df, state_df, region_df, tm_df, ts_df = preprocess_mix_feats(
        news_df, users_df)
    logger.info("Salvando arquivos de mix_feats...")
    dfs = {
        "mix_feats/mix_df.parquet": mix_df,
        "mix_feats/gap_feats.parquet": gap_df,
        "mix_feats/state_feats.parquet": state_df,
        "mix_feats/region_feats.parquet": region_df,
        "mix_feats/theme_main_feats.parquet": tm_df,
        "mix_feats/theme_sub_feats.parquet": ts_df,
    }
    for rel_file, df in dfs.items():
        file_path = os.path.join(data_path, "features", rel_file)
        _save_df_parquet(df, file_path)
    logger.info("mix_df: %d linhas", mix_df.shape[0])
    logger.info("gap_df: %d linhas | %d usuários", gap_df.shape[0],
                gap_df["userId"].nunique())
    logger.info("state_df: %d linhas | %d usuários", state_df.shape[0],
                state_df["userId"].nunique())
    logger.info("region_df: %d linhas | %d usuários", region_df.shape[0],
                region_df["userId"].nunique())
    logger.info("tm_df: %d linhas | %d usuários", tm_df.shape[0],
                tm_df["userId"].nunique())
    logger.info("ts_df: %d linhas | %d usuários", ts_df.shape[0],
                ts_df["userId"].nunique())
    return mix_df, gap_df, state_df, region_df, tm_df, ts_df


def _assemble_and_save_suggested_feats(data_path: str, mix_df, state_df,
                                       region_df, tm_df, ts_df):
    """
    Monta suggested_feats e salva os dados.

    Args:
        data_path (str): Caminho base.
        mix_df, state_df, region_df, tm_df, ts_df: DataFrames.

    Returns:
        pd.DataFrame: Dados de suggested_feats.
    """
    logger.info("Montando suggested_feats...")
    suggested = generate_suggested_feats(mix_df, state_df, region_df, tm_df, ts_df)
    file_path = os.path.join(data_path, "features", "suggested_feats.parquet")
    _save_df_parquet(suggested, file_path)
    logger.info("Suggested_feats: %d linhas | %d usuários | %d páginas",
                suggested.shape[0], suggested["userId"].nunique(),
                suggested["pageId"].nunique() if "pageId" in suggested.columns else -1)
    return suggested


def _preprocess_and_save_target(data_path: str, users_df, gap_df):
    """
    Processa o target e salva os dados.

    Args:
        data_path (str): Caminho base.
        users_df: Dados dos usuários.
        gap_df: Dados do gap.

    Returns:
        pd.DataFrame: Dados do target.
    """
    logger.info("Processando target...")
    target_df = preprocess_target(users_df, gap_df)
    file_path = os.path.join(data_path, "features", "target.parquet")
    _save_df_parquet(target_df, file_path)
    logger.info("target_df: %d linhas | %d usuários | %d páginas",
                target_df.shape[0], target_df["userId"].nunique(),
                target_df["pageId"].nunique() if "pageId" in target_df.columns else -1)
    return target_df


def _assemble_and_save_final_feats(data_path: str, suggested, target_df):
    """
    Junta suggested_feats com target e salva o conjunto final.

    Args:
        data_path (str): Caminho base.
        suggested: Dados de suggested_feats.
        target_df: Dados do target.
    """
    logger.info("Montando conjunto final de features...")
    final_feats = suggested.merge(target_df, on=["userId", "pageId"])
    file_path = os.path.join(data_path, "features", "final_feats_with_target.parquet")
    _save_df_parquet(final_feats, file_path)
    logger.info("Final_feats: %d linhas | %d usuários | %d páginas",
                final_feats.shape[0], final_feats["userId"].nunique(),
                final_feats["pageId"].nunique() if "pageId" in final_feats.columns else -1)


def pre_process_data() -> None:
    """
    Executa o pipeline de feature engineering.

    Returns:
        None
    """
    data_path = _get_data_path()
    logger.info("Dados serão salvos em: %s", data_path)
    users_df = _preprocess_and_save_users(data_path)
    selected_pageIds = list(users_df["pageId"].unique())
    news_df = _preprocess_and_save_news(data_path, selected_pageIds)
    mix_dfs = _preprocess_and_save_mix_feats(data_path, news_df, users_df)
    suggested = _assemble_and_save_suggested_feats(data_path, *mix_dfs[1:])
    target_df = _preprocess_and_save_target(data_path, users_df, mix_dfs[1])
    _assemble_and_save_final_feats(data_path, suggested, target_df)
    logger.info("Pipeline concluído com sucesso!")


if __name__ == "__main__":
    pre_process_data()
