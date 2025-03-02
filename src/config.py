import os
import logging
import colorama
from typing import Any, Dict, Tuple

import mlflow
import pandas as pd
import yaml
from dotenv import load_dotenv

colorama.init(autoreset=True)


def configure_logger(logger_name: str = __name__, level: int = logging.INFO) -> logging.Logger:
    """
    Configura e retorna um logger.

    Args:
        logger_name (str): Nome do logger.
        level (int): Nível de log.

    Returns:
        logging.Logger: Logger configurado.
    """
    logger = logging.getLogger(logger_name)
    if not logger.hasHandlers():
        handler = logging.StreamHandler()
        green = "\033[92m"
        reset = "\033[0m"
        fmt = f"{green}%(asctime)s - %(name)s - %(levelname)s - %(filename)s - %(message)s{reset}"
        handler.setFormatter(logging.Formatter(fmt))
        logger.addHandler(handler)
    logger.setLevel(level)
    return logger


logger = configure_logger()
load_dotenv()


def load_config() -> Tuple[str, Dict[str, Any]]:
    """
    Carrega a configuração com base na variável de ambiente 'ENV'.

    Nova versão que lê os arquivos de configuração diretamente do sistema de arquivos
    em vez de usar importlib.resources.

    Returns:
        tuple: (ambiente, configuração)
    """
    env = os.getenv("ENV", "dev")
    logger.info("Ambiente: %s", env)

    # Procura por arquivos de configuração em vários locais possíveis
    config_paths = [
        os.path.join(os.getcwd(), "src", "configs", f"{env}.yaml"),  # ./src/configs/env.yaml
        os.path.join(
            os.path.dirname(__file__), "configs", f"{env}.yaml"
        ),  # ./src/configs/env.yaml
        os.path.join(os.getcwd(), "configs", f"{env}.yaml"),  # ./configs/env.yaml (fallback)
        # /app/src/configs/env.yaml (no container)
        os.path.join("/app", "src", "configs", f"{env}.yaml"),
        # /app/configs/env.yaml (fallback no container)
        os.path.join("/app", "configs", f"{env}.yaml"),
    ]

    config = None
    for config_path in config_paths:
        if os.path.exists(config_path):
            logger.info("Usando arquivo de configuração: %s", config_path)
            try:
                with open(config_path, "r") as file:
                    config = yaml.safe_load(file)
                break
            except Exception as e:
                logger.warning("Erro ao carregar configuração de %s: %s", config_path, e)

    if config is None:
        logger.error("Nenhum arquivo de configuração encontrado para o ambiente %s", env)
        raise FileNotFoundError("Arquivo de configuração não encontrado")

    return env, config


ENV, CONFIG = load_config()


def get_config(key: str, default: Any = None) -> Any:
    """
    Retorna o valor de uma chave na configuração.

    Args:
        key (str): Chave de configuração.
        default (Any): Valor padrão.

    Returns:
        Any: Valor da configuração ou default.
    """
    return CONFIG.get(key, default)


def get_project_root() -> str:
    """
    Retorna o diretório raiz do projeto.

    Returns:
        str: Caminho absoluto da raiz do projeto.
    """
    return os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))


def get_data_path() -> str:
    """
    Retorna o caminho base para armazenamento de dados.
    Se USE_S3 for False, garante que o caminho seja absoluto a partir da raiz do
    projeto; caso contrário, usa o caminho configurado para S3.

    Returns:
        str: Caminho base para os dados.
    """
    use_s3 = get_config("USE_S3", False)
    if use_s3:
        data_path = get_config("S3_BUCKET", "fiap-mleng-datathon-data-grupo57/")
    else:
        data_path = get_config("LOCAL_DATA_PATH", "data/")
        # Se o caminho não for absoluto, prefixa com a raiz do projeto.
        if not os.path.isabs(data_path):
            data_path = os.path.join(get_project_root(), data_path)
    if not data_path.endswith("/"):
        data_path += "/"
    return data_path


def get_storage_mode() -> bool:
    """
    Determina se o armazenamento deve usar S3 ou local.

    Returns:
        bool: True para S3, False para local.
    """
    return get_config("USE_S3", False)


def configure_mlflow() -> None:
    """
    Configura o MLflow com os parâmetros da configuração.
    """
    mlflow.set_tracking_uri(get_config("MLFLOW_TRACKING_URI"))
    mlflow.set_registry_uri(get_config("MLFLOW_REGISTRY_URI"))
    mlflow.set_experiment(get_config("EXPERIMENT"))


USE_S3 = get_storage_mode()
S3_BUCKET = get_config("S3_BUCKET", "fiap-mleng-datathon-data-grupo57")
DATA_PATH = get_data_path()
COLD_START_THRESHOLD = get_config("COLD_START_THRESHOLD", 5)
SAMPLE_RATE = get_config("SAMPLE_RATE", 0.10)
NEWS_DIRECTORY = os.path.join(
    DATA_PATH, get_config("NEWS_DIRECTORY", "challenge-webmedia-e-globo-2023/itens/itens")
)
USERS_DIRECTORY = os.path.join(
    DATA_PATH, get_config("USERS_DIRECTORY", "challenge-webmedia-e-globo-2023/files/treino")
)
SCALING_RANGE = get_config("SCALING_RANGE", 100)
DT_TODAY = pd.Timestamp.today().date()
TODAY = DT_TODAY.strftime("%Y-%m-%d")
