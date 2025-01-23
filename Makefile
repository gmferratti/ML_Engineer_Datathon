install-uv:
	pipx install uv

create-env:
	uv venv --python 3.11

activate:
	.venv\Scripts\activate

activate-linux:
	source .venv/bin/activate

install-env:
	uv pip install -r pyproject.toml

e:
	uv pip install -e .

build:
	uv build

publish:
	uv publish