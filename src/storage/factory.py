from typing import Optional
from config import logger, get_config
from storage.base import BaseStorage
from storage.local import LocalStorage
from storage.s3 import S3Storage


def create_storage(use_s3: Optional[bool] = None,
                   s3_bucket: Optional[str] = None) -> BaseStorage:
    """
    Cria a instância de armazenamento (local ou S3).

    Args:
        use_s3 (bool, optional): Se True, usa S3; se None, usa configuração.
        s3_bucket (str, optional): Nome do bucket S3.

    Returns:
        BaseStorage: Instância de armazenamento.
    """
    if use_s3 is None:
        use_s3 = get_config("USE_S3", False)
    if use_s3:
        if s3_bucket is None:
            s3_bucket = get_config("S3_BUCKET", "fiap-mleng-datathon-data-grupo57")
        logger.info(f"Inicializando S3 no bucket '{s3_bucket}'")
        return S3Storage(bucket=s3_bucket)
    logger.info("Inicializando armazenamento local")
    return LocalStorage()
