import logging
import os
from importlib import resources

import mlflow
import yaml
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)

logger.setLevel(logging.INFO)


def load_config():
    env = os.getenv("ENV", "dev")
    logger.info("Ambiente: %s", env)
    config_package = "configs"
    try:
        with resources.open_text(config_package, f"{env}.yaml") as file:
            config = yaml.safe_load(file)
    except FileNotFoundError:
        raise FileNotFoundError(
            f"""Arquivo de configuracao {env}.yaml nao
             encontrado no pacote {config_package}"""
        )
    return env, config


ENV, CONFIG = load_config()


def get_config(key, default=None):
    return CONFIG.get(key, default)


def configure_mlflow():
    mlflow.set_tracking_uri(get_config("MLFLOW_TRACKING_URI"))
    mlflow.set_registry_uri(get_config("MLFLOW_REGISTRY_URI"))
    mlflow.set_experiment(get_config("EXPERIMENT"))
