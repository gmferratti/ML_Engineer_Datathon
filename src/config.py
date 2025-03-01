"""
Módulo de configuração para o projeto.
"""

import os
import logging
from importlib import resources
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIGURAÇÃO DO LOGGER
# --------------------------------------------------


def configure_logger(logger_name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Configura e retorna um logger com o nome e nível especificados.

    Args:
        logger_name (str): Nome do logger.
        level (int): Nível de log (default: logging.INFO).

    Returns:
        logging.Logger: Logger configurado.
    """
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = configure_logger()


# --------------------------------------------------
# CONFIGURAÇÃO DO AMBIENTE E CARREGAMENTO DE CONFIGURAÇÃO
# --------------------------------------------------
# Carrega as variáveis de ambiente
load_dotenv()


def load_config() -> Tuple[str, Dict[str, Any]]:
    """
    Carrega a configuração com base na variável de ambiente 'ENV'.

    Retorna:
        tuple: Uma tupla contendo o ambiente (str) e a configuração (dict).
    """
    env = os.getenv("ENV", "dev")
    logger.info("Ambiente: %s", env)
    config_package = "configs"
    config_file = f"{env}.yaml"

    try:
        with resources.open_text(config_package, config_file) as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        error_msg = f"""
            Arquivo de configuração '{config_file}' não encontrado no pacote '{config_package}'.
            """
        raise FileNotFoundError(error_msg)
    return env, config


ENV, CONFIG = load_config()


def get_config(key: str, default: Any = None) -> Any:
    """
    Retorna o valor de uma chave na configuração.

    Args:
        key (str): Chave de configuração.
        default (Any): Valor padrão caso a chave não exista.

    Returns:
        Any: Valor da configuração ou o valor default.
    """
    return CONFIG.get(key, default)


def get_data_path() -> str:
    """
    Retorna o caminho base para armazenamento de dados.

    Prioriza o caminho configurado em CONFIG, caso contrário usa o caminho relativo 'data/'.

    Returns:
        str: Caminho base para os dados.
    """
    # Verifica se existe uma configuração para DATA_PATH
    data_path = get_config("DATA_PATH", "data/")

    # Garante que o caminho termina com uma barra
    if not data_path.endswith("/"):
        data_path = f"{data_path}/"

    return data_path


def get_storage_mode() -> bool:
    """
    Determina se o armazenamento deve usar S3 ou sistema de arquivos local.

    Returns:
        bool: True para usar S3, False para sistema de arquivos local.
    """
    # Verifica a configuração de USE_S3, com padrão False
    return get_config("USE_S3", False)


def configure_mlflow() -> None:
    """
    Configura o MLflow utilizando os parâmetros definidos na configuração.
    """
    mlflow.set_tracking_uri(get_config("MLFLOW_TRACKING_URI"))
    mlflow.set_registry_uri(get_config("MLFLOW_REGISTRY_URI"))
    mlflow.set_experiment(get_config("EXPERIMENT"))


# --------------------------------------------------
# CONFIGURAÇÕES GERAIS
# --------------------------------------------------
USE_S3 = get_storage_mode()
S3_BUCKET = get_config("S3_BUCKET", "fiap-mleng-datathon-data-grupo57")
DATA_PATH = get_data_path()
COLD_START_THRESHOLD = get_config("COLD_START_THRESHOLD", 5)
SAMPLE_RATE = get_config("SAMPLE_RATE", 0.10)
NEWS_DIRECTORY = get_config("NEWS_DIRECTORY", "challenge-webmedia-e-globo-2023/itens/itens")
USERS_DIRECTORY = get_config("USERS_DIRECTORY", "challenge-webmedia-e-globo-2023/files/treino")

SCALING_RANGE = get_config("SCALING_RANGE", 100)

DT_TODAY = pd.Timestamp.today().date()
TODAY = DT_TODAY.strftime("%Y-%m-%d")
