from abc import ABC, abstractmethod
from typing import Any, Optional, List
import pandas as pd


class BaseStorage(ABC):
    """
    Interface para operações de armazenamento.
    """
    @abstractmethod
    def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Lê um arquivo Parquet.

        Args:
            path (str): Caminho do arquivo.
            **kwargs: Parâmetros para pd.read_parquet.

        Returns:
            pd.DataFrame: Dados lidos.
        """
        pass

    @abstractmethod
    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Salva um DataFrame como Parquet.

        Args:
            df (pd.DataFrame): Dados a salvar.
            path (str): Caminho para salvar.
            **kwargs: Parâmetros para df.to_parquet.
        """
        pass

    @abstractmethod
    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Lê um arquivo CSV.

        Args:
            path (str): Caminho do arquivo.
            **kwargs: Parâmetros para pd.read_csv.

        Returns:
            pd.DataFrame: Dados lidos.
        """
        pass

    @abstractmethod
    def write_csv(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Salva um DataFrame como CSV.

        Args:
            df (pd.DataFrame): Dados a salvar.
            path (str): Caminho para salvar.
            **kwargs: Parâmetros para df.to_csv.
        """
        pass

    @abstractmethod
    def exists(self, path: str) -> bool:
        """
        Verifica se um arquivo existe.

        Args:
            path (str): Caminho do arquivo.

        Returns:
            bool: True se existir, False caso contrário.
        """
        pass

    @abstractmethod
    def save_pickle(self, obj: Any, path: str) -> None:
        """
        Salva um objeto em pickle.

        Args:
            obj (Any): Objeto a salvar.
            path (str): Caminho para salvar.
        """
        pass

    @abstractmethod
    def load_pickle(self, path: str) -> Any:
        """
        Carrega um objeto pickle.

        Args:
            path (str): Caminho do arquivo.

        Returns:
            Any: Objeto carregado.
        """
        pass

    @abstractmethod
    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """
        Lista arquivos em um diretório.

        Args:
            path (str): Diretório.
            pattern (str, optional): Padrão para filtrar.

        Returns:
            List[str]: Lista de arquivos.
        """
        pass
