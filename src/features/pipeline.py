import os
from src.config import logger, USE_S3, DATA_PATH
from src.features.pp_mix import preprocess_mix_feats, generate_suggested_feats
from src.features.pp_news import preprocess_news
from src.features.pp_target import preprocess_target
from src.features.pp_users import preprocess_users
from src.storage.io import Storage


def _save_df_parquet(df, file_path: str, storage: Storage) -> None:
    """
    Salva um DataFrame como Parquet e loga um resumo do salvamento.
    """
    storage.write_parquet(df, file_path)
    rel_path = os.path.relpath(file_path)
    logger.info("💾 Arquivo salvo: %s | Linhas: %d", rel_path, len(df))


def _preprocess_and_save_news(data_path: str, selected_pageIds: list, storage: Storage):
    """
    Pré-processa notícias e salva o resultado.
    """
    logger.info("📰 [News] Iniciando pré-processamento das notícias...")
    news_df = preprocess_news(selected_pageIds)
    news_path = os.path.join(data_path, "features", "news_feats.parquet")
    _save_df_parquet(news_df, news_path, storage)
    logger.info(
        "📰 [News] Finalizado: %d linhas, %d páginas únicas",
        news_df.shape[0],
        news_df["pageId"].nunique(),
    )
    return news_df


def _preprocess_and_save_users(data_path: str, storage: Storage):
    """
    Pré-processa usuários e salva os dados.
    """
    logger.info("👥 [Users] Iniciando pré-processamento dos usuários...")
    users_df = preprocess_users()
    users_path = os.path.join(data_path, "features", "users_feats.parquet")
    _save_df_parquet(users_df, users_path, storage)
    logger.info(
        "👥 [Users] Concluído: %d linhas, %d páginas, %d usuários",
        users_df.shape[0],
        users_df["pageId"].nunique(),
        users_df["userId"].nunique(),
    )
    return users_df


def _preprocess_and_save_mix_feats(data_path: str, news_df, users_df, storage: Storage):
    """
    Gera mix_feats combinando notícias e usuários e salva os subconjuntos.
    """
    logger.info("🔀 [Mix] Gerando mix_feats...")
    mix_df, gap_df, state_df, region_df, tm_df, ts_df = preprocess_mix_feats(news_df, users_df)

    logger.info("🔀 [Mix] Salvando subconjuntos de mix_feats...")
    files = {
        "mix_feats/mix_df.parquet": mix_df,
        "mix_feats/gap_feats.parquet": gap_df,
        "mix_feats/state_feats.parquet": state_df,
        "mix_feats/region_feats.parquet": region_df,
        "mix_feats/theme_main_feats.parquet": tm_df,
        "mix_feats/theme_sub_feats.parquet": ts_df,
    }
    for rel_file, df in files.items():
        file_path = os.path.join(data_path, "features", rel_file)
        _save_df_parquet(df, file_path, storage)

    logger.info("🔢 [Mix] mix_df: %d linhas", mix_df.shape[0])
    logger.info(
        "👥 [Mix] gap_df: %d linhas | %d usuários",
        gap_df.shape[0],
        gap_df["userId"].nunique() if "userId" in gap_df.columns else 0,
    )
    logger.info(
        "🗺️ [Mix] state_df: %d linhas | %d usuários",
        state_df.shape[0],
        state_df["userId"].nunique() if "userId" in state_df.columns else 0,
    )
    logger.info(
        "🗺️ [Mix] region_df: %d linhas | %d usuários",
        region_df.shape[0],
        region_df["userId"].nunique() if "userId" in region_df.columns else 0,
    )
    logger.info(
        "🎨 [Mix] theme_main_feats: %d linhas | %d usuários",
        tm_df.shape[0],
        tm_df["userId"].nunique() if "userId" in tm_df.columns else 0,
    )
    logger.info(
        "🎨 [Mix] theme_sub_feats: %d linhas | %d usuários",
        ts_df.shape[0],
        ts_df["userId"].nunique() if "userId" in ts_df.columns else 0,
    )
    return mix_df, gap_df, state_df, region_df, tm_df, ts_df


def _assemble_and_save_suggested_feats(
    data_path: str, mix_df, state_df, region_df, tm_df, ts_df, storage: Storage
):
    """
    Monta suggested_feats combinando os subconjuntos e salva o resultado.
    """
    logger.info("📐 [Suggested] Montando suggested_feats...")
    suggested = generate_suggested_feats(mix_df, state_df, region_df, tm_df, ts_df)
    file_path = os.path.join(data_path, "features", "suggested_feats.parquet")
    _save_df_parquet(suggested, file_path, storage)
    logger.info(
        "📐 [Suggested] Concluído: %d linhas | %d usuários | %d páginas",
        suggested.shape[0],
        suggested["userId"].nunique() if "userId" in suggested.columns else 0,
        suggested["pageId"].nunique() if "pageId" in suggested.columns else 0,
    )
    return suggested


def _preprocess_and_save_target(data_path: str, users_df, gap_df, storage: Storage):
    """
    Processa o target e salva os dados.
    """
    logger.info("🎯 [Target] Processando target...")
    target_df = preprocess_target(users_df, gap_df)
    file_path = os.path.join(data_path, "features", "target.parquet")
    _save_df_parquet(target_df, file_path, storage)
    logger.info(
        "🎯 [Target] Concluído: %d linhas | %d usuários | %d páginas",
        target_df.shape[0],
        target_df["userId"].nunique() if "userId" in target_df.columns else 0,
        target_df["pageId"].nunique() if "pageId" in target_df.columns else 0,
    )
    return target_df


def _assemble_and_save_final_feats(data_path: str, suggested, target_df, storage: Storage):
    """
    Junta suggested_feats com target para formar o conjunto final e salva os dados.
    """
    logger.info("🔗 [Final] Montando conjunto final de features...")
    final_feats = suggested.merge(target_df, on=["userId", "pageId"])
    file_path = os.path.join(data_path, "features", "final_feats_with_target.parquet")
    _save_df_parquet(final_feats, file_path, storage)
    logger.info(
        "🔗 [Final] Concluído: %d linhas | %d usuários | %d páginas",
        final_feats.shape[0],
        final_feats["userId"].nunique() if "userId" in final_feats.columns else 0,
        final_feats["pageId"].nunique() if "pageId" in final_feats.columns else 0,
    )


def pre_process_data() -> None:
    """
    Executa o pipeline de feature engineering.
    """
    logger.info("=== 🚀 Iniciando Pipeline de Feature Engineering ===")
    logger.info("📂 Diretório base: %s", DATA_PATH)
    storage = Storage(use_s3=USE_S3)

    users_df = _preprocess_and_save_users(DATA_PATH, storage)
    selected_pageIds = list(users_df["pageId"].unique())
    news_df = _preprocess_and_save_news(DATA_PATH, selected_pageIds, storage)
    mix_df, gap_df, state_df, region_df, tm_df, ts_df = _preprocess_and_save_mix_feats(
        DATA_PATH, news_df, users_df, storage
    )
    suggested = _assemble_and_save_suggested_feats(
        DATA_PATH, mix_df, state_df, region_df, tm_df, ts_df, storage
    )
    target_df = _preprocess_and_save_target(DATA_PATH, users_df, gap_df, storage)
    _assemble_and_save_final_feats(DATA_PATH, suggested, target_df, storage)

    logger.info("=== ✅ Pipeline de Feature Engineering Finalizado com Sucesso! ===")


if __name__ == "__main__":
    pre_process_data()
