import os
import pandas as pd
from typing import Optional
from storage.io import Storage
from src.config import USE_S3, DATA_PATH, logger


def concatenate_csv_files(directory_path: str) -> pd.DataFrame:
    """
    Concatena todos os arquivos CSV de um diretório.

    Args:
        directory_path (str): Caminho dos CSVs.

    Returns:
        pd.DataFrame: Dados concatenados.
    """
    storage = Storage(use_s3=USE_S3)
    df_concat = pd.DataFrame()
    files_processed = 0
    try:
        csv_files = storage.list_files(directory_path, "*.csv")
        logger.info("📂 [Utils] Encontrados %d arquivos CSV em: %s", len(csv_files), directory_path)
        for file_path in csv_files:
            try:
                df = storage.read_csv(file_path)
                logger.info("📄 [Utils] Processado: %s | Linhas: %d | Colunas: %d",
                            os.path.basename(file_path), len(df), len(df.columns))
                df_concat = pd.concat([df_concat, df], ignore_index=True)
                files_processed += 1
            except Exception as e:
                logger.error("🚨 [Utils] Erro ao processar %s: %s", file_path, e)
    except Exception as e:
        logger.error("🚨 [Utils] Erro ao listar arquivos em %s: %s", directory_path, e)
    
    if files_processed == 0:
        logger.warning("⚠️ [Utils] Nenhum CSV encontrado em: %s", directory_path)
    
    result_df = df_concat.reset_index(drop=True)
    logger.info("🔗 [Utils] Linhas após concatenação: %d", len(result_df))
    return result_df


def ensure_directory(path: str) -> str:
    """
    Garante que o diretório do caminho fornecido exista.

    Args:
        path (str): Caminho do diretório.

    Returns:
        str: O mesmo caminho.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    logger.info("📁 [Utils] Diretório verificado: %s", os.path.dirname(path))
    return path


def get_full_path(relative_path: str) -> str:
    """
    Retorna o caminho completo com base em DATA_PATH.

    Args:
        relative_path (str): Caminho relativo.

    Returns:
        str: Caminho completo.
    """
    full_path = os.path.join(DATA_PATH, relative_path)
    logger.info("🔗 [Utils] Caminho completo gerado: %s", full_path)
    return full_path


def save_dataframe(df: pd.DataFrame, path: str, storage: Optional[Storage] = None) -> None:
    """
    Salva um DataFrame como Parquet e loga o resultado.

    Args:
        df (pd.DataFrame): Dados a salvar.
        path (str): Caminho relativo para salvar.
        storage (Storage, optional): Instância para I/O.
    """
    if storage is None:
        storage = Storage(use_s3=USE_S3)
    full_path = get_full_path(path)
    storage.write_parquet(df, full_path)
    logger.info("💾 [Utils] DataFrame salvo em: %s | Linhas: %d", full_path, len(df))


def load_dataframe(path: str, storage: Optional[Storage] = None) -> pd.DataFrame:
    """
    Carrega um DataFrame de um arquivo Parquet e loga o resultado.

    Args:
        path (str): Caminho relativo do arquivo.
        storage (Storage, optional): Instância para I/O.

    Returns:
        pd.DataFrame: Dados carregados.
    """
    if storage is None:
        storage = Storage(use_s3=USE_S3)
    full_path = get_full_path(path)
    df = storage.read_parquet(full_path)
    logger.info("📂 [Utils] DataFrame carregado de: %s | Linhas: %d", full_path, len(df))
    return df
