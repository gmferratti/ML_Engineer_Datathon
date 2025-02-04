install-uv:
	pipx install uv

create-env:
	uv venv --python 3.9

activate:
	.venv\Scripts\activate

activate-linux:
	source .venv/bin/activate

install-env:
	uv pip install -r pyproject.toml

e:
	uv pip install -e .

pp_features:
	uv run src/features/pipeline.py

build:
	uv build

publish:
	uv publish