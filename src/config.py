import os
import logging
from importlib import resources
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv

# --------------------------------------------------
# CONFIGURAÇÕES GERAIS
# --------------------------------------------------
FLAG_REMOTE = False  # Define se os dados serão processados localmente ou remotamente
COLD_START_THRESHOLD = 5  # Limite para classificar usuários como "cold start"
SAMPLE_RATE = 0.10  # Fração de amostragem dos dados

# Data de referência (cálculo)
DT_TODAY = pd.Timestamp.today().date()
TODAY = DT_TODAY.strftime("%Y-%m-%d")

# --------------------------------------------------
# CAMINHOS PARA ARMAZENAMENTO DE DADOS
# --------------------------------------------------
LOCAL_DATA_PATH = (
    "C:/Users/gufer/OneDrive/Documentos/FIAP/Fase_05/ML_Engineer_Datathon/data/"
)
REMOTE_DATA_PATH = "s3://..."  # Caminho remoto (S3, GCS, etc.)

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
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
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
        error_msg = (
            f"Arquivo de configuração '{config_file}' não encontrado no pacote '{config_package}'."
        )
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


def configure_mlflow() -> None:
    """
    Configura o MLflow utilizando os parâmetros definidos na configuração.
    """
    mlflow.set_tracking_uri(get_config("MLFLOW_TRACKING_URI"))
    mlflow.set_registry_uri(get_config("MLFLOW_REGISTRY_URI"))
    mlflow.set_experiment(get_config("EXPERIMENT"))
