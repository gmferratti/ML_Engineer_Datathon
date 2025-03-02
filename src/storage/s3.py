import os
import tempfile
import pickle
import fnmatch
from typing import Any, Optional, List, BinaryIO
import boto3
import pandas as pd
from botocore.exceptions import ClientError
from src.config import logger
from .base import BaseStorage


class S3UploadFile:
    """
    Auxilia o upload de arquivos para S3.
    """

    def __init__(self, temp_file: BinaryIO, s3_client, bucket: str, key: str):
        self.temp_file = temp_file
        self.temp_name = temp_file.name
        self.s3_client = s3_client
        self.bucket = bucket
        self.key = key  # Já normalizado
        self.closed = False

    def write(self, data: bytes) -> int:
        return self.temp_file.write(data)

    def close(self) -> None:
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

    def _normalize_key(self, key: str) -> str:
        """
        Garante que o separador seja "/", remove duplicações
        e retira o prefixo do bucket se existir.
        """
        # 1) Troca "\" por "/"
        key = key.replace("\\", "/")
        # 2) Se começar com "bucket/...", remove esse prefixo
        if key.startswith(self.s3_bucket + "/"):
            key = key[len(self.s3_bucket) + 1 :]
        # Remove barras iniciais redundantes
        key = key.lstrip("/")
        return key

    def read_parquet(self, path: str, **kwargs) -> pd.DataFrame:
        norm_key = self._normalize_key(path)
        return pd.read_parquet(f"s3://{self.s3_bucket}/{norm_key}", **kwargs)

    def write_parquet(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        norm_key = self._normalize_key(path)
        df.to_parquet(f"s3://{self.s3_bucket}/{norm_key}", **kwargs)
        logger.info(f"Arquivo salvo em s3://{self.s3_bucket}/{norm_key}")

    def read_csv(self, path: str, **kwargs) -> pd.DataFrame:
        norm_key = self._normalize_key(path)
        return pd.read_csv(f"s3://{self.s3_bucket}/{norm_key}", **kwargs)

    def write_csv(self, df: pd.DataFrame, path: str, **kwargs) -> None:
        norm_key = self._normalize_key(path)
        df.to_csv(f"s3://{self.s3_bucket}/{norm_key}", **kwargs)
        logger.info(f"Arquivo salvo em s3://{self.s3_bucket}/{norm_key}")

    def exists(self, path: str) -> bool:
        norm_key = self._normalize_key(path)
        try:
            self.s3_client.head_object(Bucket=self.s3_bucket, Key=norm_key)
            return True
        except ClientError as e:
            if e.response["Error"]["Code"] == "404":
                return False
            raise

    def _get_s3_file(self, path: str, mode: str) -> BinaryIO:
        norm_key = self._normalize_key(path)
        if mode.startswith("r"):
            temp = tempfile.NamedTemporaryFile(delete=False)
            self.s3_client.download_fileobj(self.s3_bucket, norm_key, temp)
            temp.close()
            return open(temp.name, mode)
        elif mode.startswith("w"):
            temp = tempfile.NamedTemporaryFile(delete=False)
            return S3UploadFile(temp, self.s3_client, self.s3_bucket, norm_key)
        raise ValueError(f"Modo não suportado: {mode}")

    def save_pickle(self, obj: Any, path: str) -> None:
        norm_key = self._normalize_key(path)
        with self._get_s3_file(norm_key, "w") as f:
            pickle.dump(obj, f)
        logger.info(f"Objeto salvo em s3://{self.s3_bucket}/{norm_key}")

    def load_pickle(self, path: str) -> Any:
        norm_key = self._normalize_key(path)
        try:
            with self._get_s3_file(norm_key, "r") as f:
                return pickle.load(f)
        except Exception as e:
            logger.error(f"Erro ao carregar s3://{self.s3_bucket}/{norm_key}: {e}")
            raise

    def list_files(self, path: str, pattern: Optional[str] = None) -> List[str]:
        norm_key = self._normalize_key(path)
        try:
            response = self.s3_client.list_objects_v2(Bucket=self.s3_bucket, Prefix=norm_key)
            if "Contents" not in response:
                return []
            files = []
            for obj in response["Contents"]:
                key = obj["Key"]
                if pattern is None or self._match_pattern(key, pattern):
                    files.append(key)
            return files
        except Exception as e:
            logger.error(f"Erro ao listar s3://{self.s3_bucket}/{norm_key}: {e}")
            raise

    def _match_pattern(self, filename: str, pattern: str) -> bool:
        return fnmatch.fnmatch(os.path.basename(filename), pattern)
