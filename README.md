# ML_Engineer_Datathon

[![MIT License](https://img.shields.io/badge/License-MIT-green.svg)](https://choosealicense.com/licenses/mit/)

Este documento está localizado em **docs/** e reflete a estrutura do projeto, que espelha a organização dos módulos contidos na pasta *src/*.

---

## Índice

1. [Objetivo e Contexto](#objetivo-e-contexto)
2. [Visão Geral do Projeto](#visão-geral-do-projeto)
3. [Fluxo de Execução](#fluxo-de-execução)
4. [Configuração e Ambiente](#configuração-e-ambiente)
5. [Empacotamento e Deploy](#empacotamento-e-deploy)
6. [Estrutura do Projeto](#estrutura-do-projeto)
7. [Endpoints e Monitoramento](#endpoints-e-monitoramento)
8. [Contribuição e Notas](#contribuição-e-notas)
9. [Referências](#referências)

---

## Objetivo e Contexto

Desenvolver um sistema de recomendação personalizado, com foco em prever a próxima notícia a ser lida por um usuário com base no consumo de notícias do G1. O sistema é projetado para lidar tanto com usuários com histórico consolidado quanto com aqueles em situação de *cold start*.

**Integrantes:**

- Antonio Eduardo de Oliveira Lima
- Gustavo Mendonça Ferratti
- Luiz Claudio Santana Barbosa
- Mauricio de Araujo Pintor
- Rodolfo Olivieri

---

## Visão Geral do Projeto

O ML_Engineer_Datathon é composto por diversos módulos que, integrados, formam um sistema completo de recomendação. Cada módulo possui documentação específica, mas aqui apresentamos um resumo dos principais componentes:

- **Feature Engineering:**  
  Processamento dos dados brutos de notícias e usuários, extração e transformação de features e cálculo do score de engajamento (TARGET).

- **Treinamento e Ranking:**  
  Utilização do **LightGBMRanker** com o objetivo *lambdarank* para treinar o modelo e gerar a ordenação dos itens, otimizando métricas como o NDCG.

- **API de Predição:**  
  Implementada com FastAPI, esta API processa os inputs, trata casos de *cold start* e retorna recomendações ordenadas com os metadados relevantes.

- **Avaliação:**  
  Pipeline que utiliza a métrica **NDCG@10** para mensurar a qualidade do ranking gerado pelo modelo.

---

## Fluxo de Execução

1. **Pré-processamento dos Dados:**  
   - **Notícias:** Consolidação, filtragem e extração de informações (localidade, temas, data/hora).  
   - **Usuários:** Processamento de históricos, extração de features temporais e identificação de *cold start*.  
   - **Integração:** Combinação dos dados e cálculo do TARGET com base em cliques, tempo na página, scroll, recência e outras variáveis.  
     
     A fórmula utilizada é:

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
     
     Valores negativos são ajustados, aplicando transformações como `log1p` e escalonamento via Min-Max Scaling.

2. **Treinamento e Geração de Ranking:**  
   - O modelo **LightGBMRanker** é treinado para otimizar a ordenação dos itens com base no NDCG.
   - Durante a predição, as features são combinadas para definir a ordem das recomendações.

3. **API de Predição:**  
   - A API processa requisições, tratando os inputs e gerando respostas diferenciadas para casos de *cold start* ou histórico de consumo consolidado.

4. **Avaliação:**  
   - Um pipeline calcula o **NDCG@10** para medir a eficácia do ranking gerado, permitindo ajustes e melhorias contínuas.

---

## Configuração e Ambiente

### Principais Comandos

Todos os comandos para execução dos módulos estão definidos no **Makefile**. Entre eles:

- `make evaluate`
- `make predict`
- `make train`
- Entre outros.

### Variáveis de Ambiente

Crie um arquivo `.env` na raiz do projeto contendo, no mínimo:

```
ENV="dev"
```


Os possíveis valores para `ENV` são `"dev"`, `"staging"` ou `"prod"`.

### Credenciais e Servidores

- **Azure:** Configure as credenciais necessárias no `.env` para acesso aos recursos.
- **MLflow:** Inicie o servidor do MLflow (por exemplo, com `mlflow ui`) utilizando a URI apropriada.
- **API:** Execute a API conforme especificado no Makefile (por exemplo, via `uvicorn`).

---

## Empacotamento e Deploy

Após testes e validação, o sistema é empacotado com Docker e passa por um rigoroso processo de deploy:

- **Docker:**  
  - **Dockerfile:** Define o container otimizado para produção.
  - **docker-compose.yml:** Orquestra os serviços para ambiente local.

- **Deploy:**  
  Documentação adicional sobre o deploy está disponível em `DEPLOY_AWS.md`, que detalha o processo para a AWS utilizando ECS Fargate.

---

## 🏗️ Estrutura do Projeto

```
.
├── Dockerfile              # Configuração de container otimizado
├── docker-compose.yml      # Orquestração de serviços Docker
├── run-local.sh            # Script de inicialização simplificada
├── DEPLOY_AWS.md           # Instruções para deploy na AWS
├── pyproject.toml          # Dependências do projeto
├── configs/                # Configurações de ambiente
└── src/                    # Código-fonte
    ├── api/                # API de recomendação
    ├── data/               # Manipulação de dados
    ├── evaluation/         # Métricas e avaliação
    ├── features/           # Feature engineering
    ├── predict/            # Pipeline de predição
    ├── recommendation_model/# Modelos de recomendação
    ├── storage/            # Abstração de armazenamento
    └── train/              # Pipeline de treinamento
```

## Endpoints e Monitoramento

### Endpoints da API

- **`GET /health`**: Verifica a integridade da API.
- **`GET /info`**: Retorna informações sobre o modelo e o ambiente.
- **`POST /predict`**: Gera recomendações para um usuário.  

  **Exemplo de requisição:**

  ```json
  {
    "userId": "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297",
    "max_results": 5,
    "minScore": 0.3
  }
  ```

### Monitoramento de Performance

A resposta da API inclui métricas detalhadas, como:
  ```json
      {
        "processing_time_ms": 123.45,
        "timing_details": {
          "dependencies": 0.01,
          "prediction": 0.12,
          "formatting": 0.01,
          "total_ms": 123.45
        }
      }
  ```
Essas informações auxiliam na identificação e resolução de gargalos.

---



# Setups e Rodagens em Containers


## ✨ Principais Características

- **API FastAPI:** Interface de alta performance para recomendações em tempo real
- **MLflow Integration:** Registro e versionamento de modelos com tracking de métricas
- **Containerização:** Deploy simplificado via Docker
- **Otimizações de Performance:** Caching, profiling e redução de processamento redundante
- **Suporte a Ambientes:** Configurações para desenvolvimento local e produção
- **Suporte a Cold Start:** Tratamento para usuários novos ou com pouca informação
- **Portabilidade AWS:** Pronto para deploy em AWS ECS Fargate

## 🚀 Inicialização Rápida

### Pré-requisitos

- Docker
- Docker Compose
- Bash (para o script de inicialização)

### Executar Localmente (Desenvolvimento)

```bash
# Tornar o script executável
chmod +x run-local.sh

# Iniciar todos os serviços em modo desenvolvimento (API + MLflow)
./run-local.sh

# Para reconstruir as imagens (após alterações no código)
./run-local.sh dev full rebuild

# Para ver logs após inicialização
./run-local.sh dev full logs
```

### Executar em Modo Produção (Usando MLflow remoto)

```bash
# Configurar credenciais AWS para acesso ao S3 (opcional)
export AWS_ACCESS_KEY_ID="sua-chave"  
export AWS_SECRET_ACCESS_KEY="seu-secret"

# Iniciar apenas a API em modo produção
./run-local.sh prod api
```

### Opções de Inicialização

```bash
# Ver ajuda e todas as opções disponíveis
./run-local.sh help

# Exemplos comuns:
./run-local.sh dev api      # Apenas API em modo desenvolvimento
./run-local.sh dev mlflow   # Apenas MLflow local
./run-local.sh prod api     # API em modo produção (MLflow remoto)
```



## 🔍 Endpoints da API

- **`GET /health`**: Verifica a saúde da API
- **`GET /info`**: Informações sobre o modelo e ambiente
- **`POST /predict`**: Gera recomendações para um usuário

### Exemplo de requisição para `/predict`:

```json
{
  "userId": "4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297",
  "max_results": 5,
  "minScore": 0.3
}
```

## 📊 Monitoramento de Performance

A API agora inclui métricas detalhadas de performance para ajudar a identificar e resolver gargalos. Ao fazer uma chamada para `/predict`, a resposta incluirá:

```json
{
  "processing_time_ms": 123.45,
  "timing_details": {
    "dependencies": 0.01,
    "prediction": 0.12,
    "formatting": 0.01,
    "total_ms": 123.45
  }
}
```

## 🐳 Configuração Docker

O projeto utiliza dois arquivos principais:

- **`Dockerfile`**: Container otimizado para produção
- **`docker-compose.yml`**: Configuração de ambiente local

## 🌐 Configuração de Ambientes

O sistema suporta dois ambientes principais:

### 1. Desenvolvimento (`dev`)

- MLflow local para experimentos
- Armazenamento local de dados
- Taxa de amostragem reduzida para testes rápidos

### 2. Produção (`prod`)

- MLflow remoto para registro de modelos
- Armazenamento S3 para dados e artefatos
- Taxa de amostragem completa para melhor performance


## 🚢 Deploy na AWS

Para fazer o deploy do sistema na AWS ECS usando Fargate:

1. Siga as instruções detalhadas em `DEPLOY_AWS.md`
2. Automatize o processo com o script de deploy incluído



### Contribuindo

Para colaborar com o projeto:

1. Clone o repositório.
2. Instale as dependências com `uv pip install -e .`.
3. Crie uma nova branch para suas alterações.
4. Envie um Pull Request com uma descrição clara das mudanças.

---

### Notas

- O script `run-local.sh` deve ser executado a partir da raiz do projeto.
- Certifique-se de manter a consistência nas configurações de ambiente e documentação entre os módulos.

---


## Referências

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)  
- [Lambdarank Paper](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)  
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)  
- [Pandas Documentation](https://pandas.pydata.org/docs/)  
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

