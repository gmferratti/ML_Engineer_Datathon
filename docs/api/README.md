# API – News Recommender

Este módulo é responsável por disponibilizar a API do News Recommender, permitindo a predição de recomendações de notícias em tempo real. Utilizando o framework FastAPI, a API integra o carregamento dos dados (via data_loader), a utilização do modelo de recomendação treinado (por exemplo, LightGBMRanker) e o processamento dos inputs para gerar respostas com as recomendações para um usuário.

---

## Índice

- [Visão Geral](#visão-geral)
- [Fluxo de Execução](#fluxo-de-execução)
- [Configuração e Dependências](#configuração-e-dependências)
- [Referências](#referências)

---

## Visão Geral

A API do News Recommender oferece serviços de predição e monitoramento do modelo de recomendação. Quando uma requisição de predição é realizada, a API:

1. Carrega os dados pré-processados (features de notícias e de usuários) utilizando o módulo `data_loader.py`.
2. Carrega o modelo treinado (por exemplo, LightGBMRanker) via MLflow ou de armazenamento local.
3. Processa o input recebido (por exemplo, um `userId`) e constrói o DataFrame de features esperado pelo modelo.
4. Executa a predição e gera uma lista de recomendações, com tratamento especial para casos de **cold start** (usuários sem histórico).
5. Retorna uma resposta estruturada com as recomendações, a versão do modelo, uma flag indicando se o usuário é cold start e o tempo de processamento.

Além disso, a API inclui endpoints para monitoramento e obtenção de informações sobre o status do modelo e do ambiente de execução.

---

## Fluxo de Execução

1. **Inicialização:**  
   - A API é criada utilizando FastAPI e configura middlewares (como CORS).
   - Durante o startup, o modelo e os dados de predição são carregados, utilizando funções definidas em `data_loader.py` e `core.py`.

2. **Processamento de Requisições:**  
   - Ao receber uma requisição de predição, a API extrai os parâmetros do request, constrói o input para o modelo e invoca o método `predict`.
   - Se o usuário não for encontrado (indicando cold start), o pipeline adapta a resposta retornando as notícias mais recentes.
   - O tempo de processamento é medido e incluído na resposta.

3. **Monitoramento:**  
   - Endpoints para saúde (health) e informações (info) permitem monitorar o status da API e do modelo, auxiliando na manutenção e na identificação de problemas.

---

## Configuração e Dependências

- **Framework:**  
  A API foi desenvolvida utilizando [FastAPI](https://fastapi.tiangolo.com/).

- **Integração com MLflow:**  
  O modelo é carregado via MLflow, utilizando a URI configurada (por exemplo, `http://localhost:5001`).

- **Armazenamento de Dados:**  
  Os dados são lidos a partir de arquivos (como Parquet) utilizando o módulo `storage.io`. As variáveis `DATA_PATH` e `USE_S3` são definidas no arquivo de configuração (`src/config.py`).

- **Logs:**  
  A aplicação utiliza o módulo de logging para registrar o fluxo de execução e eventuais erros.

---

## Referências

- [FastAPI Documentation](https://fastapi.tiangolo.com/)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Python Logging](https://docs.python.org/3/library/logging.html)

---

Este README.md fornece uma visão detalhada do funcionamento da API de predição do News Recommender, explicando o fluxo de execução, a integração com o modelo e as configurações essenciais. Para mais informações, consulte os READMEs dos demais módulos (como Feature Engineering, Treinamento, Avaliação, etc.).
