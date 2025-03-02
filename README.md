# ML_Engineer_Datathon

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

Último desafio FIAP para a conclusão da pós em Machine Learning Engineering.

---

## Objetivo

Desenvolver um sistema de recomendação personalizado baseado no consumo de notícias do G1, com o objetivo de prever qual será a próxima notícia lida por um usuário. O sistema é capaz de lidar com diversos perfis de usuários – desde aqueles com amplo histórico de consumo até os que estão acessando o site pela primeira vez (cold start).

---

## Integrantes

- Antonio Eduardo de Oliveira Lima
- Gustavo Mendonça Ferratti
- Luiz Claudio Santana Barbosa
- Mauricio de Araujo Pintor
- Rodolfo Olivieri

---

## Visão Geral do Projeto

O projeto ML_Engineer_Datathon é composto por vários módulos que, juntos, formam um sistema completo de recomendação. Cada módulo possui sua própria documentação detalhada e é responsável por uma parte específica do fluxo. Os módulos principais são:

- **Feature Engineering:**  
  Processa os dados brutos (notícias e usuários), extrai e transforma as features e calcula o score de engajamento (TARGET).

- **Treinamento e Ranking:**  
  Utiliza o **LightGBMRanker** – um modelo de ranking baseado no LightGBM com o objetivo **lambdarank** – para treinar o modelo e gerar uma ordenação dos itens. O treinamento otimiza métricas de ranking, como o NDCG.

- **API de Predição:**  
  Disponibiliza uma API (via FastAPI) para realizar predições em tempo real. A API carrega os dados pré-processados e o modelo treinado, trata os inputs (inclusive para casos de cold start) e retorna as recomendações ordenadas com metadados.

- **Avaliação:**  
  Um pipeline de avaliação utiliza a métrica **NDCG@10** para medir a qualidade do ranking gerado pelo modelo, comparando os scores preditos com os valores reais de engajamento.

---

## Fluxo Geral

1. **Pré-processamento dos Dados:**  
   - **Notícias:** Concatenação, filtragem e extração de informações (localidade, temas, data/hora de publicação).
   - **Usuários:** Processamento de históricos, extração de features temporais e definição da flag `coldStart`.
   - **Mix de Features:** Combinação dos dados de notícias e usuários, cálculo de gaps temporais e proporções por categoria.
   - **Cálculo do TARGET:** Cálculo do score de engajamento usando métricas como cliques, tempo na página, scroll, recência, tamanho do histórico e gap temporal.  
     
     A fórmula geral é:
     ```
     scoreBase = numberOfClicksHistory 
                 + 1.5 * (timeOnPageHistory / 1000)
                 + scrollPercentageHistory 
                 - (minutesSinceLastVisit / 60)
     ```
     Em seguida:
     ```
     rawScore = scoreBase * (historySize / 130) * (1 / (1 + (timeGapDays / 50)))
     ```
     Valores negativos são ajustados para zero, aplicando `log1p`, escalonamento via Min-Max Scaling e arredondamento para gerar o TARGET final.

2. **Treinamento e Ranking:**  
   - O **LightGBMRanker** é treinado com os dados processados utilizando o objetivo **lambdarank**, que otimiza a ordenação dos itens com base em métricas de ranking como o NDCG.
   - Durante a predição, as features dos clientes e das notícias são combinadas para gerar scores, que determinam a ordem dos itens recomendados.

3. **API de Predição:**  
   - A API, desenvolvida com FastAPI, recebe requisições contendo parâmetros como `userId`, `max_results` e `minScore`.
   - Caso o usuário não seja encontrado (indicando cold start), a API retorna as notícias mais recentes com score definido como "desconhecido", juntamente com metadados (título, URL, issuedDate e issuedTime).
   - Caso o usuário possua histórico, o modelo gera scores que determinam o ranking final das notícias.

4. **Avaliação:**  
   - Um pipeline de avaliação calcula o **NDCG@10** para medir a qualidade do ranking.  
     - **NDCG@10:** É uma métrica que avalia a ordenação dos itens. Um valor de 1 indica uma ordenação perfeita, enquanto valores próximos de 0 indicam uma ordenação ruim.

---

## Configuração Inicial e Ambiente

- **Principais Comandos:**  
  Todos os comandos principais para executar os módulos do projeto estão definidos no **Makefile**. Consulte o Makefile para comandos como:
  - `make evaluate`
  - `make predict`
  - `make train`
  - Entre outros.

- **Variáveis de Ambiente:**  
  É necessário criar um arquivo `.env` na raiz do projeto com, pelo menos, a seguinte variável:
```
  ENV = "dev"
```
Os valores possíveis para `ENV` são `"dev"`, `"staging"` ou `"prod"`.

- **Credenciais e Servidores:**  
- **Azure:** Certifique-se de configurar as credenciais da Azure no arquivo `.env` para acesso aos recursos necessários.
- **MLflow:** Inicie o servidor do MLflow (por exemplo, via `mlflow ui`) com a URI configurada (ex.: `http://localhost:5001`).
- **API:** Inicie a API utilizando o comando especificado no Makefile (por exemplo, via `uvicorn`).

---

## Empacotamento e Deploy

Após a finalização do desenvolvimento e testes, o sistema é empacotado utilizando Docker e passa por etapas de validação e deploy. Detalhes sobre o empacotamento e deploy estão documentados nos respectivos READMEs dos módulos de API e Docker.

---

## Referências e Documentação Complementar

Para obter mais informações sobre cada etapa do projeto, consulte os READMEs específicos dos módulos:

- **Feature Engineering:** Documentação em `README_data_loader.md` e outros arquivos de pré-processamento.
- **Treinamento e Ranking:** Documentação em `README_ranker.md`.
- **API de Predição:** Documentação em `README_app.md`.
- **Avaliação:** Documentação em `README_predict.md`.

Outras referências importantes:
- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Lambdarank Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

---

Este README.md global oferece uma visão integrada do projeto ML_Engineer_Datathon, resumindo as principais etapas, módulos e configurações necessárias para a execução e deploy do sistema. Para detalhes específicos de cada componente, consulte os READMEs individuais disponíveis em cada pasta do projeto.
