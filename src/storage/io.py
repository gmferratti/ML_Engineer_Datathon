from typing import Optional, List, Any
import pandas as pd
from src.config import get_config, USE_S3
from .base import BaseStorage
from .factory import create_storage


class Storage(BaseStorage):
    """
    Fachada para acesso a arquivos delegando para o backend correto.
    """

    def __init__(self, use_s3: Optional[bool] = None):
        """
        Inicializa o armazenamento.

        Args:
            use_s3 (bool, optional): Se True, usa S3. Se None, usa configuração.
        """
        self._storage = create_storage(use_s3=use_s3)
        self.use_s3 = USE_S3 if use_s3 is None else use_s3
        self.s3_bucket = (
            get_config("S3_BUCKET", "fiap-mleng-datathon-data-grupo57") if self.use_s3 else None
        )

    def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Lê um arquivo Parquet.

        Args:
            path (str): Caminho do arquivo.
            **kwargs: Parâmetros para pd.read_parquet.

        Returns:
            pd.DataFrame: Dados lidos.
        """
        return self._storage.read_parquet(path, **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Salva um DataFrame como Parquet.

        Args:
            df (pd.DataFrame): Dados a salvar.
            path (str): Caminho para salvar.
            **kwargs: Parâmetros para df.to_parquet.
        """
        self._storage.write_parquet(df, path, **kwargs)

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Lê um arquivo CSV.

        Args:
            path (str): Caminho do arquivo.
            **kwargs: Parâmetros para pd.read_csv.

        Returns:
            pd.DataFrame: Dados lidos.
        """
        return self._storage.read_csv(path, **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Salva um DataFrame como CSV.

        Args:
            df (pd.DataFrame): Dados a salvar.
            path (str): Caminho para salvar.
            **kwargs: Parâmetros para df.to_csv.
        """
        self._storage.write_csv(df, path, **kwargs)

    def exists(self, path: str) -> bool:
        """
        Verifica se um arquivo existe.

        Args:
            path (str): Caminho do arquivo.

        Returns:
            bool: True se existir, False caso contrário.
        """
        return self._storage.exists(path)

    def save_pickle(self, obj: Any, path: str) -> None:
        """
        Salva um objeto em pickle.

        Args:
            obj (Any): Objeto a salvar.
            path (str): Caminho para salvar.
        """
        self._storage.save_pickle(obj, path)

    def load_pickle(self, path: str) -> Any:
        """
        Carrega um objeto pickle.

        Args:
            path (str): Caminho do arquivo.

        Returns:
            Any: Objeto carregado.
        """
        return self._storage.load_pickle(path)

    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """
        Lista arquivos em um diretório.

        Args:
            path (str): Diretório.
            pattern (str, optional): Padrão para filtrar.

        Returns:
            List[str]: Lista de arquivos.
        """
        return self._storage.list_files(path, pattern)
