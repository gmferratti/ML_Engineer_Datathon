#!/usr/bin/env python
"""
Script para transferir dados do ambiente local para o S3.
Mantém a estrutura de diretórios para o pipeline.
"""

import logging
from pathlib import Path
from typing import List
import boto3

# Configurações fixas
S3_BUCKET = "fiap-mleng-datathon-data-grupo57"
NEWS_DIRECTORY = "challenge-webmedia-e-globo-2023/itens/itens"
USERS_DIRECTORY = "challenge-webmedia-e-globo-2023/files/treino"

LOCAL_SOURCE_DIR = Path("notebooks/mauricio/data")

# Mapeamento de diretórios locais para o S3
PATH_MAPPING = {
    "itens": NEWS_DIRECTORY,
    "treino": USERS_DIRECTORY,
}

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("upload-script")


def create_s3_client():
    """Cria e retorna um cliente S3."""
    return boto3.client("s3")


def create_bucket_if_not_exists(s3_client, bucket: str):
    """Cria o bucket se ele não existir."""
    try:
        s3_client.head_bucket(Bucket=bucket)
        logger.info("Bucket %s já existe.", bucket)
    except Exception:
        logger.info("Bucket %s não encontrado. Criando bucket.", bucket)
        try:
            s3_client.create_bucket(Bucket=bucket)
            logger.info("Bucket %s criado com sucesso.", bucket)
        except Exception as e:
            logger.error("Erro ao criar bucket %s: %s", bucket, e)
            raise


def list_csv_files(base_dir: Path, subfolder: str) -> List[Path]:
    """Lista arquivos CSV num subdiretório."""
    folder = base_dir / subfolder
    if not folder.exists():
        logger.warning("Diretório não encontrado: %s", folder)
        return []
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() == ".csv"
    ])


def upload_file(s3_client, local_file: Path, target_prefix: str) -> bool:
    """Envia o arquivo CSV para o S3 mantendo o mesmo nome."""
    filename = local_file.name
    s3_key = f"{target_prefix}/{filename}"
    try:
        logger.info("Enviando %s para s3://%s/%s",
                    filename, S3_BUCKET, s3_key)
        s3_client.upload_file(str(local_file), S3_BUCKET, s3_key)
        logger.info("Arquivo %s enviado com sucesso.", filename)
        return True
    except Exception as e:
        logger.error("Erro ao enviar %s: %s", filename, e)
        return False


def main():
    logger.info("Iniciando upload de arquivos para o S3")
    s3_client = create_s3_client()
    create_bucket_if_not_exists(s3_client, S3_BUCKET)
    for local_dir, s3_dir in PATH_MAPPING.items():
        logger.info("Processando diretório: %s", local_dir)
        files = list_csv_files(LOCAL_SOURCE_DIR, local_dir)
        logger.info("Encontrados %d arquivos CSV em %s",
                    len(files), local_dir)
        for file in files:
            upload_file(s3_client, file, s3_dir)
    logger.info("Upload concluído.")


if __name__ == "__main__":
    main()
