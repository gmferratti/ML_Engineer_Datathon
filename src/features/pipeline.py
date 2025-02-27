"""Módulo principal para execução do pipeline de feature engineering."""

import os

from config import logger
from feat_settings import FLAG_REMOTE, LOCAL_DATA_PATH, REMOTE_DATA_PATH
from features.pp_mix import preprocess_mix_feats, generate_suggested_feats
from features.pp_news import preprocess_news
from features.pp_target import preprocess_target
from features.pp_users import preprocess_users


def _get_data_path() -> str:
    """
    Retorna o caminho de dados de acordo com a flag de execução remota ou local.

    Returns:
        str: Caminho onde os dados serão salvos/carregados.
    """
    if FLAG_REMOTE:
        logger.info("Remote storage chosen!")
        return REMOTE_DATA_PATH
    logger.info("Local storage chosen!")
    return LOCAL_DATA_PATH


def _save_df_parquet(df, file_path: str) -> None:
    """
    Cria o diretório (se não existir) e salva o DataFrame em parquet no path especificado.

    Args:
        df (pandas.DataFrame): DataFrame a ser salvo.
        file_path (str): Caminho completo (com nome do arquivo) onde será salvo o parquet.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    df.to_parquet(file_path)


def _preprocess_and_save_news(data_path: str):
    """
    Executa o pré-processamento de notícias e salva o resultado em parquet.

    Args:
        data_path (str): Caminho base para salvar os arquivos.

    Returns:
        pandas.DataFrame: DataFrame com as features de notícias pré-processadas.
    """
    logger.info("Pre-processing news info...")
    news_df = preprocess_news()
    news_path = os.path.join(data_path, "features", "news_feats.parquet")
    logger.info(f"Saving pre-processed news parquet at {news_path}...")
    _save_df_parquet(news_df, news_path)
    return news_df


def _preprocess_and_save_users(data_path: str):
    """
    Executa o pré-processamento de usuários e salva o resultado em parquet.

    Args:
        data_path (str): Caminho base para salvar os arquivos.

    Returns:
        pandas.DataFrame: DataFrame com as features de usuários pré-processadas.
    """
    logger.info("Pre-processing users info...")
    users_df = preprocess_users()
    users_path = os.path.join(data_path, "features", "users_feats.parquet")
    logger.info(f"Saving pre-processed users parquet at {users_path}...")
    _save_df_parquet(users_df, users_path)
    return users_df


def _preprocess_and_save_mix_feats(data_path: str, news_df, users_df):
    """
    Gera as mix_feats combinando 'news_df' e 'users_df',
    e salva dataframes intermediários em parquet.

    Args:
        data_path (str): Caminho base para salvar os arquivos.
        news_df (pandas.DataFrame): DataFrame de notícias.
        users_df (pandas.DataFrame): DataFrame de usuários.

    Returns:
        tuple: Vários DataFrames (mix_df, gap_df, state_df, region_df, tm_df, ts_df).
    """
    logger.info("Generating mix feats...")
    mix_df, gap_df, state_df, region_df, tm_df, ts_df = preprocess_mix_feats(
        news_df, users_df
    )

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
        file_path = os.path.join(
            data_path, "features", "mix_feats", f"{file_name}.parquet"
        )
        logger.info(f"Saving {file_name} parquet at {file_path}...")
        _save_df_parquet(df, file_path)

    return mix_df, gap_df, state_df, region_df, tm_df, ts_df


def _assemble_and_save_suggested_feats(data_path: str,
                                       mix_df,
                                       state_df,
                                       region_df,
                                       tm_df,
                                       ts_df):
    """
    Monta as 'suggested_feats' a partir dos dataframes de mix e dimensões,
    e salva em formato parquet.

    Args:
        data_path (str): Caminho base para salvar os arquivos.
        mix_df (pandas.DataFrame): DataFrame mix.
        state_df (pandas.DataFrame): DataFrame de estados.
        region_df (pandas.DataFrame): DataFrame de regiões.
        tm_df (pandas.DataFrame): DataFrame de temas principais.
        ts_df (pandas.DataFrame): DataFrame de temas secundários.

    Returns:
        pandas.DataFrame: DataFrame com as features sugeridas.
    """
    logger.info("Assembling suggested features df...")
    suggested_feats = generate_suggested_feats(
        mix_df, state_df, region_df, tm_df, ts_df
    )
    suggested_feats_path = os.path.join(
        data_path, "features", "suggested_feats.parquet"
    )
    _save_df_parquet(suggested_feats, suggested_feats_path)
    logger.info(f"Suggested features DF saved at {suggested_feats_path}")

    return suggested_feats


def _preprocess_and_save_target(data_path: str, users_df, gap_df):
    """
    Executa o pré-processamento do target a partir de 'users_df' e 'gap_df',
    e salva em formato parquet.

    Args:
        data_path (str): Caminho base para salvar os arquivos.
        users_df (pandas.DataFrame): DataFrame de usuários.
        gap_df (pandas.DataFrame): DataFrame de gap (relacionado a mix).

    Returns:
        pandas.DataFrame: DataFrame com o target pré-processado.
    """
    logger.info("Pre-processing target values...")
    target_df = preprocess_target(users_df, gap_df)
    target_path = os.path.join(data_path, "features", "target.parquet")
    logger.info(f"Saving pre-processed target parquet at {target_path}...")
    _save_df_parquet(target_df, target_path)
    return target_df


def _assemble_and_save_final_feats(data_path: str, suggested_feats, target_df):
    """
    Junta as 'suggested_feats' com o 'target_df' e salva o conjunto final de
    features em formato parquet.

    Args:
        data_path (str): Caminho base para salvar os arquivos.
        suggested_feats (pandas.DataFrame): DataFrame das features sugeridas.
        target_df (pandas.DataFrame): DataFrame do target.
    """
    logger.info("Assembling final features DF with target...")
    final_feats = suggested_feats.merge(target_df, on=["userId", "pageId"])
    final_feats_path = os.path.join(
        data_path, "features", "final_feats_with_target.parquet"
    )
    _save_df_parquet(final_feats, final_feats_path)
    logger.info(f"Final features DF saved at {final_feats_path}")


def pre_process_data() -> None:
    """
    Executa o pipeline de feature engineering:
        1. Define o caminho dos dados (remoto ou local).
        2. Pré-processa as informações de news e users.
        3. Gera as mix feats combinando news e users.
        4. Propõe uma sugestão inicial de features.
        5. Cria e salva o target.
        6. Gera o conjunto final de features com target.
    """
    data_path = _get_data_path()
    logger.info(f"Info will be saved at {data_path}")

    # 1. Pré-processamento de notícias e usuários
    news_df = _preprocess_and_save_news(data_path)
    users_df = _preprocess_and_save_users(data_path)

    # 2. Criação das mix feats
    (mix_df, gap_df, state_df,
     region_df, tm_df, ts_df) = _preprocess_and_save_mix_feats(
        data_path, news_df, users_df
    )

    # 3. Geração de suggested_feats
    suggested_feats = _assemble_and_save_suggested_feats(
        data_path, mix_df, state_df, region_df, tm_df, ts_df
    )

    # 4. Criação e salvamento do target
    target_df = _preprocess_and_save_target(data_path, users_df, gap_df)

    # 5. Montagem e salvamento das features finais com target
    _assemble_and_save_final_feats(data_path, suggested_feats, target_df)

    logger.info("Pre-processing complete!")


if __name__ == "__main__":
    pre_process_data()
