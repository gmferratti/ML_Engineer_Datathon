.PHONY: setup clean lint format check-format install install-uv create-env install-env e build publish pp_features mlflow-server create-kernel remove-kernel

PYTHON = python3
LINTING_PATHS = src/ tests/
MLFLOW_PORT ?= 5001

#######################################################################################################################################################
################################################################### SETUP & CONFIG ####################################################################
#######################################################################################################################################################

# Setup inicial do projeto
setup: clean install-uv create-env install-env e create-kernel

# Instalação do uv
install-uv:
	pip install uv

# Criação do ambiente virtual
create-env:
	uv venv --python 3.9

# Instalação das dependências no ambiente
install-env:
	uv pip install -r pyproject.toml

# Instalação em modo editável
e:
	uv pip install -e .

# Limpa arquivos cache e build
clean:
	find . -type d -name "__pycache__" -exec rm -rf {} +
	find . -type d -name "*.pyc" -exec rm -rf {} +
	find . -type d -name ".pytest_cache" -exec rm -rf {} +
	find . -type d -name "*.egg-info" -exec rm -rf {} +
	find . -type d -name ".coverage" -exec rm -rf {} +
	rm -rf build/ dist/ .eggs/

# Build e publicação
build:
	uv build

publish:
	uv publish

# Linting e formatação
lint:
	uv pip install flake8
	flake8 $(LINTING_PATHS)

format:
	uv pip install black isort
	black $(LINTING_PATHS)
	isort $(LINTING_PATHS)

check-format:
	uv pip install black isort
	black --check $(LINTING_PATHS)
	isort --check-only $(LINTING_PATHS)

# MLflow
mlflow-server:
	mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlruns --host 0.0.0.0 --port $(MLFLOW_PORT)

# Jupyter
create-kernel:
	python -m ipykernel install --user --name=datathon --display-name="Python (Datathon)"

remove-kernel:
	jupyter kernelspec uninstall datathon -y

#######################################################################################################################################################
################################################################### PROJECT RUNNING ###################################################################
#######################################################################################################################################################

pp_features:
	uv run src/features/pipeline.py

# train:
# 	uv run src/train/train.py

train:
	uv run src/train/pipeline.py

predict:
	uv run src/predict/predict.py

run_all:
	uv run src/features/pipeline.py && uv run src/train/pipeline.py

local_api:
	uv run src/api/app.py

docker_api:
	docker-compose up --build