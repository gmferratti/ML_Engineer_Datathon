import os
import tempfile
import pickle
import fnmatch
from typing import Any, Optional, List, BinaryIO
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from config import logger
from storage.base import BaseStorage


class S3UploadFile:
    """
    Auxilia o upload de arquivos para S3.
    """

    def __init__(self, temp_file: BinaryIO, s3_client, bucket: str, key: str):
        self.temp_file = temp_file
        self.temp_name = temp_file.name
        self.s3_client = s3_client
        self.bucket = bucket
        self.key = key
        self.closed = False

    def write(self, data: bytes) -> int:
        """
        Escreve dados no arquivo temporário.

        Args:
            data (bytes): Dados a escrever.

        Returns:
            int: Número de bytes escritos.
        """
        return self.temp_file.write(data)

    def close(self) -> None:
        """
        Fecha o arquivo e faz upload para S3.
        """
        if not self.closed:
            self.temp_file.close()
            self.s3_client.upload_file(self.temp_name, self.bucket, self.key)
            os.unlink(self.temp_name)
            self.closed = True

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class S3Storage(BaseStorage):
    """
    Implementa armazenamento via Amazon S3.
    """

    def __init__(self, bucket: str):
        """
        Inicializa a conexão com o S3.

        Args:
            bucket (str): Nome do bucket S3.
        """
        self.s3_bucket = bucket
        try:
            self.s3_client = boto3.client("s3")
            self.s3_resource = boto3.resource("s3")
            self.s3_client.head_bucket(Bucket=self.s3_bucket)
            logger.info(f"S3 válido para bucket '{self.s3_bucket}'")
        except ClientError as e:
            code = e.response.get("Error", {}).get("Code", "Desconhecido")
            if code == "404":
                logger.error(f"Bucket '{self.s3_bucket}' não encontrado")
            elif code == "403":
                logger.error(f"Sem permissão para bucket '{self.s3_bucket}'")
            else:
                logger.error(f"Erro ao acessar S3: {e}")
            raise

    def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Lê um arquivo Parquet do S3.

        Args:
            path (str): Caminho relativo no bucket.
            **kwargs: Parâmetros para pd.read_parquet.

        Returns:
            pd.DataFrame: Dados lidos.
        """
        return pd.read_parquet(f"s3://{self.s3_bucket}/{path}", **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Salva um DataFrame como Parquet no S3.

        Args:
            df (pd.DataFrame): Dados a salvar.
            path (str): Caminho relativo no bucket.
            **kwargs: Parâmetros para df.to_parquet.
        """
        df.to_parquet(f"s3://{self.s3_bucket}/{path}", **kwargs)
        logger.info(f"Arquivo salvo em s3://{self.s3_bucket}/{path}")

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        """
        Lê um CSV do S3.

        Args:
            path (str): Caminho relativo.
            **kwargs: Parâmetros para pd.read_csv.

        Returns:
            pd.DataFrame: Dados lidos.
        """
        return pd.read_csv(f"s3://{self.s3_bucket}/{path}", **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        """
        Salva um DataFrame como CSV no S3.

        Args:
            df (pd.DataFrame): Dados a salvar.
            path (str): Caminho relativo.
            **kwargs: Parâmetros para df.to_csv.
        """
        df.to_csv(f"s3://{self.s3_bucket}/{path}", **kwargs)
        logger.info(f"Arquivo salvo em s3://{self.s3_bucket}/{path}")

    def exists(self, path: str) -> bool:
        """
        Verifica se um arquivo existe no S3.

        Args:
            path (str): Caminho relativo.

        Returns:
            bool: True se existir, False caso contrário.
        """
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=path)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def _get_s3_file(self, path: str, mode: str) -> BinaryIO:
        """
        Obtém um arquivo do S3 para leitura ou escrita.

        Args:
            path (str): Caminho relativo.
            mode (str): Modo ('rb' ou 'wb').

        Returns:
            BinaryIO: Objeto de arquivo.
        """
        if mode.startswith("r"):
            temp = tempfile.NamedTemporaryFile(delete=False)
            self.s3_client.download_fileobj(self.s3_bucket, path, temp)
            temp.close()
            return open(temp.name, mode)
        elif mode.startswith("w"):
            temp = tempfile.NamedTemporaryFile(delete=False)
            return S3UploadFile(temp, self.s3_client, self.s3_bucket, path)
        raise ValueError(f"Modo não suportado: {mode}")

    def save_pickle(self, obj: Any, path: str) -> None:
        """
        Salva um objeto em pickle no S3.

        Args:
            obj (Any): Objeto a salvar.
            path (str): Caminho relativo.
        """
        with self._get_s3_file(path, "wb") as f:
            pickle.dump(obj, f)
        logger.info(f"Objeto salvo em s3://{self.s3_bucket}/{path}")

    def load_pickle(self, path: str) -> Any:
        """
        Carrega um objeto pickle do S3.

        Args:
            path (str): Caminho relativo.

        Returns:
            Any: Objeto carregado.
        """
        try:
            with self._get_s3_file(path, "rb") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar s3://{self.s3_bucket}/{path}: {e}")
            raise

    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        """
        Lista arquivos em um prefixo do S3.

        Args:
            path (str): Prefixo.
            pattern (str, optional): Padrão para filtrar.

        Returns:
            List[str]: Lista de arquivos.
        """
        try:
            response = self.s3_client.list_objects_v2(
                Bucket=self.s3_bucket, Prefix=path)
            if "Contents" not in response:
                return []
            files = []
            for obj in response["Contents"]:
                key = obj["Key"]
                if pattern is None or self._match_pattern(key, pattern):
                    files.append(key)
            return files
        except Exception as e:
            logger.error(f"Erro ao listar s3://{self.s3_bucket}/{path}: {e}")
            raise

    def _match_pattern(self, filename: str, pattern: str) -> bool:
        """
        Verifica se o nome do arquivo corresponde ao padrão.

        Args:
            filename (str): Nome do arquivo.
            pattern (str): Padrão.

        Returns:
            bool: True se corresponder, False caso contrário.
        """
        return fnmatch.fnmatch(os.path.basename(filename), pattern)
