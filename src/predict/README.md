# Módulo de Predição – News Recommender

Este documento descreve o módulo de **Predição** do projeto News Recommender. Nele, é implementado o pipeline que, a partir dos dados pré-processados (features de notícias e usuários) e de um modelo treinado, gera recomendações de notícias para um usuário específico.

> **Observação:** Este README foca exclusivamente na lógica de predição. Informações sobre feature engineering, treinamento e API de predição podem ser encontradas nos respectivos READMEs dos outros módulos.

---

## Índice

- [Visão Geral](#visão-geral)
- [Fluxo de Predição](#fluxo-de-predição)
  - [Validação e Construção do Input](#validação-e-construção-do-input)
  - [Geração de Recomendações para Usuários Não Cold Start](#geração-de-recomendações-para-usuários-não-cold-start)
  - [Tratamento de Cold Start](#tratamento-de-cold-start)
  - [Conversão de Campos de Data e Hora](#conversão-de-campos-de-data-e-hora)
- [Integração com MLflow e Modelos](#integração-com-mlflow-e-modelos)
- [Estrutura de Constantes](#estrutura-de-constantes)
- [Referências e Documentação Complementar](#referências-e-documentação-complementar)

---

## Visão Geral

O módulo de predição tem como objetivo gerar um ranking de notícias para um usuário com base em seu histórico e nas características dos itens. Ao receber um `userId`, o pipeline:

- Carrega os dados pré-processados (features de notícias e de usuários).
- Constrói o input esperado pelo modelo combinando as features dos clientes e das notícias.
- Se o usuário não for encontrado e o `userId` possuir um formato válido (por exemplo, um hash com tamanho indicativo), assume-se que ele é cold start e o sistema retorna as notícias mais recentes, com score definido como `"desconhecido"`.
- Caso contrário, utiliza o modelo treinado para gerar scores e, então, enriquece as recomendações com metadados (título, URL, issuedDate e issuedTime).

Esta abordagem garante que, mesmo para usuários sem histórico, o sistema possa fornecer recomendações relevantes baseadas na novidade das notícias.

---

## Fluxo de Predição

### Validação e Construção do Input

- **Função `validate_features`:**  
  Verifica se as colunas esperadas estão presentes nos DataFrames de clientes e notícias.

- **Função `build_model_input`:**  
  Obtém as features do cliente (usando `get_client_features`) e das notícias. Em seguida, replica as features do cliente para cada notícia e organiza as colunas no formato esperado pelo modelo.

### Geração de Recomendações para Usuários Não Cold Start

- Se o usuário for encontrado no DataFrame de clientes, o pipeline:
  1. Monta o input final para o modelo.
  2. Executa a predição utilizando o modelo treinado (carregado via MLflow ou local).
  3. Utiliza a função `_generate_normal_recommendations` para mapear os scores gerados às notícias correspondentes, enriquecendo cada recomendação com metadados (título, URL, issuedDate e issuedTime).

### Tratamento de Cold Start

- **Condição de Cold Start:**  
  Se o usuário não for encontrado (ou seja, as features não estão disponíveis) e o `userId` tiver tamanho compatível com um hash válido (ex.: comprimento ≥ 64), o sistema assume que o usuário é cold start.

- **Fluxo Cold Start:**  
  O pipeline utiliza a função `_generate_cold_start_recommendations`, que:
  - Ordena as notícias a partir de uma coluna combinada de data e hora (obtida a partir dos campos `issuedDate` e `issuedTime`).
  - Seleciona as notícias mais recentes (de forma decrescente).
  - Define o score fixo como `"desconhecido"` para cada recomendação.
  - Retorna as recomendações enriquecidas com os metadados extraídos.

---

## Integração com MLflow e Modelos

O pipeline de predição utiliza:
- **`load_data_for_prediction`** para carregar os DataFrames pré-processados.
- **`load_model_from_mlflow`** para carregar o modelo treinado (geralmente um LightGBMRanker ou outro modelo registrado via MLflow).

Essas integrações garantem que o pipeline de predição utilize dados atualizados e o modelo mais recente registrado no sistema.

---