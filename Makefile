.PHONY: setup clean lint format check-format install install-uv create-env install-env e build publish pp_features mlflow-server create-kernel remove-kernel

PYTHON = python3
LINTING_PATHS = src/ tests/
MLFLOW_PORT ?= 5001
LOCAL_HOST = 127.0.0.1 # ou 0.0.0.0

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

# Jupyter
create-kernel:
	python -m ipykernel install --user --name=datathon --display-name="Python (Datathon)"

remove-kernel:
	jupyter kernelspec uninstall datathon -y

#######################################################################################################################################################
################################################################### MLFLOW LOCAL ######################################################################
#######################################################################################################################################################

.PHONY: setup-mlflow

setup-mlflow:
	@echo "Criando diretório mlruns (se não existir)..."
	mkdir -p mlruns
	@echo "Configurando permissões para mlruns..."
	chmod -R 777 mlruns
	@echo "Verificando a existência de mlflow.db..."
	[ -f mlflow.db ] || (touch mlflow.db && chmod 666 mlflow.db)

mlflow-start: setup-mlflow
	mlflow server --host $(LOCAL_HOST) --port $(MLFLOW_PORT) 



#######################################################################################################################################################
################################################################### PROJECT RUNNING ###################################################################
#######################################################################################################################################################

.PHONY: pp_features train predict evaluate run

pp_features:
	PYTHONPATH="." uv run src/features/pipeline.py

train:
	PYTHONPATH="." uv run src/train/pipeline.py

predict:
	PYTHONPATH="." uv run src/predict/pipeline.py

evaluate:
	PYTHONPATH="." uv run src/evaluation/pipeline.py
	
run: pp_features train predict # evaluate

local_api:
	PYTHONPATH="." uvicorn src.api.app:app --reload

docker_api:
	PYTHONPATH="." docker-compose up --build

test:
	PYTHONPATH="." pytest --disable-warnings

# IMPORTANTE: Para rodar estes comandos no Windows: 

# 1. instale o MakeFile usando choco install make no VSCode (modo Admin) 
# 2. Garanta que você esteja usando o GitBash com PYTHONPATH configurado corretamente.
# 3. Caso esteja salvando as informações em algum Drive, evite usar o hard link do UV
# Para isso, configure: export UV_LINK_MODE=copy. Irá ficar mais lento, mas pelo menos
