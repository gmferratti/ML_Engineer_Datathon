import os
import pathlib
import pickle
from typing import Any, Optional, List
import pandas as pd
from config import logger
from storage.base import BaseStorage


class LocalStorage(BaseStorage):
    """
    Implementa armazenamento local via sistema de arquivos.
    """

    def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Lê um arquivo Parquet local.

        Args:
            path (str): Caminho do arquivo.
            **kwargs: Parâmetros para pd.read_parquet.

        Returns:
            pd.DataFrame: Dados lidos.
        """
        return pd.read_parquet(path, **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Salva um DataFrame como Parquet localmente.

        Args:
            df (pd.DataFrame): Dados a salvar.
            path (str): Caminho para salvar.
            **kwargs: Parâmetros para df.to_parquet.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, **kwargs)
        rel_path = os.path.relpath(path)
        logger.info(f"Arquivo salvo: {rel_path}")

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Lê um arquivo CSV local.

        Args:
            path (str): Caminho do arquivo.
            **kwargs: Parâmetros para pd.read_csv.

        Returns:
            pd.DataFrame: Dados lidos.
        """
        return pd.read_csv(path, **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Salva um DataFrame como CSV localmente.

        Args:
            df (pd.DataFrame): Dados a salvar.
            path (str): Caminho para salvar.
            **kwargs: Parâmetros para df.to_csv.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, **kwargs)
        rel_path = os.path.relpath(path)
        logger.info(f"Arquivo salvo: {rel_path}")

    def exists(self, path: str) -> bool:
        """
        Verifica se um arquivo existe localmente.

        Args:
            path (str): Caminho do arquivo.

        Returns:
            bool: True se existir, False caso contrário.
        """
        return os.path.exists(path)

    def save_pickle(self, obj: Any, path: str) -> None:
        """
        Salva um objeto em formato pickle.

        Args:
            obj (Any): Objeto a salvar.
            path (str): Caminho para salvar.
        """
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        rel_path = os.path.relpath(path)
        logger.info(f"Objeto salvo: {rel_path}")

    def load_pickle(self, path: str) -> Any:
        """
        Carrega um objeto pickle.

        Args:
            path (str): Caminho do arquivo.

        Returns:
            Any: Objeto carregado.
        """
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar {path}: {e}")
            raise

    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """
        Lista arquivos em um diretório.

        Args:
            path (str): Diretório.
            pattern (str, optional): Padrão para filtrar.

        Returns:
            List[str]: Lista de arquivos.
        """
        try:
            if not os.path.isdir(path):
                return []
            p = pathlib.Path(path)
            if pattern:
                return [str(f) for f in p.glob(pattern)]
            return [str(f) for f in p.iterdir() if f.is_file()]
        except Exception as e:
            logger.error(f"Erro ao listar em {path}: {e}")
            raise
