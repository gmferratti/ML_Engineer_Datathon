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

        logger.info("Encontrados %d CSVs em %s", len(csv_files), directory_path)
        for file_path in csv_files:
            try:
                df = storage.read_csv(file_path)
                logger.info(
                    "Arquivo: %s, linhas: %d, cols: %d",
                    os.path.basename(file_path),
                    len(df),
                    len(df.columns),
                )
                df_concat = pd.concat([df_concat, df])
                files_processed += 1
            except Exception as e:
                logger.error("Erro ao processar %s: %s", file_path, e)
    except Exception as e:
        logger.error("Erro ao listar em %s: %s", directory_path, e)
    if files_processed == 0:
        logger.warning("Nenhum CSV encontrado em %s", directory_path)
    result_df = df_concat.reset_index(drop=True)
    logger.info("Linhas após concatenação: %d", len(result_df))
    return result_df


def ensure_directory(path: str) -> str:
    """
    Garante que o diretório existe.

    Args:
        path (str): Caminho do diretório.

    Returns:
        str: Mesmo caminho.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    return path


def get_full_path(relative_path: str) -> str:
    """
    Retorna o caminho completo baseado em DATA_PATH.

    Args:
        relative_path (str): Caminho relativo.

    Returns:
        str: Caminho completo.
    """
    return os.path.join(DATA_PATH, relative_path)


def save_dataframe(df: pd.DataFrame, path: str, storage: Optional[Storage] = None) -> None:
    """
    Salva um DataFrame como Parquet.

    Args:
        df (pd.DataFrame): Dados a salvar.
        path (str): Caminho relativo para salvar.
        storage (Storage, optional): Instância para I/O.
    """
    if storage is None:
        storage = Storage(use_s3=USE_S3)
    full_path = get_full_path(path)
    storage.write_parquet(df, full_path)
    logger.info("DataFrame salvo em: %s, linhas: %d", full_path, len(df))


def load_dataframe(path: str, storage: Optional[Storage] = None) -> pd.DataFrame:
    """
    Carrega um DataFrame de um arquivo Parquet.

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
    logger.info("DataFrame carregado de: %s, linhas: %d", full_path, len(df))
    return df
