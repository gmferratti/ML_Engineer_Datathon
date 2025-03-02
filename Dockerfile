FROM python:3.9-slim

WORKDIR /app

# Instalação de dependências de sistema necessárias
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*


# Instalação dos pacotes necessários
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r pyproject.toml && \
    rm -rf /root/.cache/pip /tmp/pip-* /root/.cache/uv

# Copia arquivos de código
COPY src ./src

# Define variáveis de ambiente
ENV PYTHONPATH=/app
ENV ENV=prod
ENV PYTHONUNBUFFERED=1

# Expõe a porta da API
EXPOSE 8000

# Comando para iniciar a API
CMD ["uvicorn", "src.api.app:app", "--host", "0.0.0.0", "--port", "8000"]