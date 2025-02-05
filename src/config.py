import os
from dotenv import load_dotenv
import yaml

load_dotenv()

ENV = os.getenv("ENV", "dev")
config_file = os.path.join("src", "configs", f"{ENV}.yaml")
with open(config_file, "r") as f:
    CONFIG = yaml.safe_load(f)
