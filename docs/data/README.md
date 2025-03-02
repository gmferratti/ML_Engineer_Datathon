# Data Loader – News Recommender

Este módulo é responsável por carregar e preparar os dados utilizados nas etapas de predição e avaliação do modelo de recomendação. Ele integra diversas tarefas essenciais, tais como:

- **Carregamento de Dados:**  
  Lê os arquivos de features completos (por exemplo, X_train_full.parquet) e os arquivos de avaliação (X_test.parquet e y_test.parquet) a partir do diretório definido pela configuração (DATA_PATH).

- **Preparação dos DataFrames:**  
  Separa os dados em DataFrames distintos para notícias e usuários. No caso das notícias, há a possibilidade de realizar um merge com metadados (como título, URL, issuedDate e issuedTime) se a flag `include_metadata` for definida como `True`.

- **Extração de Features Específicas:**  
  Fornece utilitários para:
  - Obter as características de um usuário específico.
  - Identificar as notícias que o usuário ainda não visualizou.

- **Carregamento do Modelo:**  
  Inclui uma função para carregar o modelo treinado (por exemplo, um LightGBMRanker) a partir de um arquivo pickle, para que possa ser utilizado nas etapas de predição e avaliação.

## Visão Geral

O módulo `data_loader.py` assegura que os dados sejam lidos e preparados de forma consistente, permitindo que as demais partes do sistema (como a predição e avaliação) possam utilizar DataFrames com o formato e os tipos corretos. As configurações de diretórios e uso de armazenamento (local ou S3) são controladas através das variáveis definidas em `src/config.py`.

## Configurações e Dependências

- **Configurações:**  
  As variáveis como DATA_PATH e USE_S3 são definidas em `src/config.py` e utilizadas para localizar os arquivos de dados.
  
- **Dependências:**  
  O módulo utiliza a biblioteca `pandas` para manipulação de dados, juntamente com os utilitários de armazenamento definidos em `storage.io`.

## Notas Adicionais

- Durante o carregamento dos dados para predição, o campo `pageId` é convertido para string para garantir a compatibilidade.
- Se a opção `include_metadata` estiver ativada, o módulo realiza um merge dos dados de notícias com os metadados disponíveis no arquivo `data/features/news_feats.parquet`. Caso haja problemas no merge, um aviso é gerado e os dados sem metadados serão utilizados.
- O módulo também oferece funções para extrair características específicas dos usuários e identificar quais notícias ainda não foram visualizadas, facilitando a construção do input para o modelo.

## Referências

- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Python Logging](https://docs.python.org/3/library/logging.html)
