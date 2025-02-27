"""Módulo principal para execução do pipeline de feature engineering."""

import os
from config import logger, FLAG_REMOTE, LOCAL_DATA_PATH, REMOTE_DATA_PATH
from features.pp_mix import preprocess_mix_feats, generate_suggested_feats
from features.pp_news import preprocess_news
from features.pp_target import preprocess_target
from features.pp_users import preprocess_users


def _get_data_path() -> str:
    """
    Retorna o caminho de dados de acordo com a flag de execução remota ou local.
    """
    if FLAG_REMOTE:
        logger.info("Armazenamento remoto selecionado.")
        return REMOTE_DATA_PATH
    logger.info("Armazenamento local selecionado.")
    return LOCAL_DATA_PATH


def _save_df_parquet(df, file_path: str) -> None:
    """
    Cria o diretório (se não existir) e salva o DataFrame em formato Parquet.
    """
    directory = os.path.dirname(file_path)
    os.makedirs(directory, exist_ok=True)
    df.to_parquet(file_path)
    # Exibe apenas o caminho relativo para maior clareza
    rel_path = os.path.relpath(file_path)
    logger.info("Arquivo salvo: %s", rel_path)


def _preprocess_and_save_news(data_path: str, selected_pageIds: list):
    """
    Pré-processa notícias e salva o resultado em Parquet.
    """
    logger.info("Iniciando pré-processamento das notícias...")
    news_df = preprocess_news(selected_pageIds)
    news_path = os.path.join(data_path, "features", "news_feats.parquet")
    _save_df_parquet(news_df, news_path)
    logger.info("Notícias pré-processadas: %s linhas | %d páginas únicas",
                news_df.shape[0], news_df["pageId"].nunique())
    return news_df


def _preprocess_and_save_users(data_path: str):
    """
    Pré-processa usuários e salva o resultado em Parquet.
    """
    logger.info("Iniciando pré-processamento dos usuários...")
    users_df = preprocess_users()
    users_path = os.path.join(data_path, "features", "users_feats.parquet")
    _save_df_parquet(users_df, users_path)
    # Exibe a informação resumida apenas uma vez
    logger.info("Usuários: %d linhas | %d páginas únicas | %d usuários únicos",
                users_df.shape[0], users_df["pageId"].nunique(), users_df["userId"].nunique())
    return users_df


def _preprocess_and_save_mix_feats(data_path: str, news_df, users_df):
    """
    Gera as mix_feats combinando news_df e users_df e salva DataFrames intermediários em Parquet.
    """
    logger.info("Gerando mix_feats...")
    mix_df, gap_df, state_df, region_df, tm_df, ts_df = preprocess_mix_feats(news_df, users_df)

    logger.info("Salvando arquivos intermediários de mix_feats...")
    dfs_to_save = {
        "mix_feats/mix_df.parquet": mix_df,
        "mix_feats/gap_feats.parquet": gap_df,
        "mix_feats/state_feats.parquet": state_df,
        "mix_feats/region_feats.parquet": region_df,
        "mix_feats/theme_main_feats.parquet": tm_df,
        "mix_feats/theme_sub_feats.parquet": ts_df,
    }
    for rel_file, df in dfs_to_save.items():
        file_path = os.path.join(data_path, "features", rel_file)
        _save_df_parquet(df, file_path)

    logger.info("mix_df: %s linhas", mix_df.shape[0])
    logger.info("gap_df: %s linhas | %d usuários únicos", gap_df.shape[0], gap_df["userId"].nunique())
    logger.info("state_df: %s linhas | %d usuários únicos", state_df.shape[0], state_df["userId"].nunique())
    logger.info("region_df: %s linhas | %d usuários únicos", region_df.shape[0], region_df["userId"].nunique())
    logger.info("tm_df: %s linhas | %d usuários únicos", tm_df.shape[0], tm_df["userId"].nunique())
    logger.info("ts_df: %s linhas | %d usuários únicos", ts_df.shape[0], ts_df["userId"].nunique())

    return mix_df, gap_df, state_df, region_df, tm_df, ts_df


def _assemble_and_save_suggested_feats(data_path: str, mix_df, state_df, region_df, tm_df, ts_df):
    """
    Monta as suggested_feats a partir dos DataFrames de mix e dimensões e salva em Parquet.
    """
    logger.info("Montando suggested_feats...")
    suggested_feats = generate_suggested_feats(mix_df, state_df, region_df, tm_df, ts_df)
    rel_file = "suggested_feats.parquet"
    suggested_feats_path = os.path.join(data_path, "features", rel_file)
    _save_df_parquet(suggested_feats, suggested_feats_path)
    logger.info("Suggested_feats: %s linhas | %d usuários únicos | %d páginas únicas",
                suggested_feats.shape[0],
                suggested_feats["userId"].nunique(),
                suggested_feats["pageId"].nunique() if "pageId" in suggested_feats.columns else -1)
    return suggested_feats


def _preprocess_and_save_target(data_path: str, users_df, gap_df):
    """
    Pré-processa o target a partir de users_df e gap_df e salva o resultado em Parquet.
    """
    logger.info("Processando target...")
    target_df = preprocess_target(users_df, gap_df)
    rel_file = "target.parquet"
    target_path = os.path.join(data_path, "features", rel_file)
    _save_df_parquet(target_df, target_path)
    logger.info("target_df: %s linhas | %d usuários únicos | %d páginas únicas",
                target_df.shape[0],
                target_df["userId"].nunique(),
                target_df["pageId"].nunique() if "pageId" in target_df.columns else -1)
    return target_df


def _assemble_and_save_final_feats(data_path: str, suggested_feats, target_df):
    """
    Junta suggested_feats com target_df e salva o conjunto final de features em Parquet.
    """
    logger.info("Montando conjunto final de features com target...")
    final_feats = suggested_feats.merge(target_df, on=["userId", "pageId"])
    rel_file = "final_feats_with_target.parquet"
    final_feats_path = os.path.join(data_path, "features", rel_file)
    _save_df_parquet(final_feats, final_feats_path)
    logger.info("Final_feats: %s linhas | %d usuários únicos | %d páginas únicas",
                final_feats.shape[0],
                final_feats["userId"].nunique(),
                final_feats["pageId"].nunique() if "pageId" in final_feats.columns else -1)
    return final_feats


def pre_process_data() -> None:
    """
    Executa o pipeline de feature engineering:
      1. Define o caminho dos dados (remoto ou local).
      2. Pré-processa as informações de usuários e notícias.
      3. Gera as mix_feats combinando news e users.
      4. Gera suggested_feats.
      5. Cria e salva o target.
      6. Monta o conjunto final de features com target.
    """
    data_path = _get_data_path()
    logger.info("Dados serão salvos em: %s", data_path)

    # Pré-processamento de usuários
    users_df = _preprocess_and_save_users(data_path)

    selected_pageIds = list(users_df["pageId"].unique())

    # Pré-processamento de notícias
    news_df = _preprocess_and_save_news(data_path, selected_pageIds)

    # Pré-processamento e salvamento das mix_feats
    mix_df, gap_df, state_df, region_df, tm_df, ts_df = _preprocess_and_save_mix_feats(data_path, news_df, users_df)

    # Geração das suggested_feats
    suggested_feats = _assemble_and_save_suggested_feats(data_path, mix_df, state_df, region_df, tm_df, ts_df)

    # Processamento e salvamento do target
    target_df = _preprocess_and_save_target(data_path, users_df, gap_df)

    # Montagem do conjunto final de features com target
    final_feats = _assemble_and_save_final_feats(data_path, suggested_feats, target_df)

    logger.info("Pipeline concluído com sucesso!")


if __name__ == "__main__":
    pre_process_data()
