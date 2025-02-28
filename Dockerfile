FROM python:3.9-slim

WORKDIR /app

# Instalação dos pacotes necessários
COPY pyproject.toml .
RUN pip install --no-cache-dir uv && \
    uv pip install --system -r pyproject.toml && \
    # Instala dependências para S3
    pip install --no-cache-dir boto3>=1.37.2 && \
    rm -rf /root/.cache/pip

# Copia arquivos de código
COPY src ./src

# Define variáveis de ambiente
ENV PYTHONPATH=/app
ENV ENV=prod

# Expõe a porta da API
EXPOSE 8000

# Comando para iniciar a API
CMD ["python", "src/api/app.py"]