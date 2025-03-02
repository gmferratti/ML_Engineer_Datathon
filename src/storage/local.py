# src/storage/local.py
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

    def _normalize_local_path(self, path: str) -> str:
        """
        Converte barras invertidas para barras normais
        e normaliza o caminho (removendo coisas como ./ e ../).
        """
        # Substitui qualquer "\" por "/" e depois normaliza
        path = path.replace("\\", "/")
        return os.path.normpath(path)

    def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        path = self._normalize_local_path(path)
        return pd.read_parquet(path, **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        path = self._normalize_local_path(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_parquet(path, **kwargs)
        rel_path = os.path.relpath(path)
        logger.info(f"Arquivo salvo: {rel_path}")

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        path = self._normalize_local_path(path)
        return pd.read_csv(path, **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        path = self._normalize_local_path(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path, **kwargs)
        rel_path = os.path.relpath(path)
        logger.info(f"Arquivo salvo: {rel_path}")

    def exists(self, path: str) -> bool:
        path = self._normalize_local_path(path)
        return os.path.exists(path)

    def save_pickle(self, obj: Any, path: str) -> None:
        path = self._normalize_local_path(path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(obj, f)
        rel_path = os.path.relpath(path)
        logger.info(f"Objeto salvo: {rel_path}")

    def load_pickle(self, path: str) -> Any:
        path = self._normalize_local_path(path)
        try:
            with open(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar {path}: {e}")
            raise

    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        path = self._normalize_local_path(path)
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
