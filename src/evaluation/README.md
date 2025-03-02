# Pipeline de Avaliação – News Recommender

Este documento descreve o pipeline de avaliação do modelo de recomendação do News Recommender. O pipeline tem como objetivo medir a performance do modelo utilizando a métrica **NDCG@10**, que é amplamente utilizada em problemas de ranking para avaliar a qualidade da ordenação dos itens recomendados.

---

## Índice

- [Visão Geral](#visão-geral)
- [Métrica NDCG@10](#métrica-ndcg10)
  - [O que é NDCG?](#o-que-é-ndcg)
  - [Interpretação do NDCG@10](#interpretação-do-ndcg10)
- [Fluxo do Pipeline de Avaliação](#fluxo-do-pipeline-de-avaliação)
- [Como Executar](#como-executar)
- [Referências](#referências)

---

## Visão Geral

O pipeline de avaliação tem como foco medir a performance do modelo de recomendação utilizando a métrica **NDCG@10** (Normalized Discounted Cumulative Gain, com corte em 10 itens). Esse pipeline utiliza um conjunto de dados de avaliação, onde cada linha contém as features do par usuário-notícia e o valor real do **TARGET** (engajamento). Com base nessas informações, o modelo gera uma pontuação para cada par e o pipeline calcula o NDCG@10 para verificar a qualidade da ordenação gerada pelo modelo.

---

## Métrica NDCG@10

### O que é NDCG?

**Normalized Discounted Cumulative Gain (NDCG)** é uma métrica usada para avaliar a eficácia de sistemas de ranking, especialmente em aplicações de busca e recomendação. Ela compara a ordem gerada pelo modelo com a ordem ideal (ou ordenação de referência) e penaliza erros de ordenação que afetam as posições mais altas da lista.

A métrica é composta por dois passos:

1. **Cumulative Gain (CG):**  
   O ganho cumulativo é a soma dos ganhos associados aos itens na ordem em que são apresentados. Cada item possui um ganho (por exemplo, a relevância do item).

2. **Discounted Cumulative Gain (DCG):**  
   O ganho cumulativo é “descontado” de acordo com a posição dos itens na lista. Itens em posições mais baixas recebem um desconto maior, pois o impacto na experiência do usuário é menor.
   
   A fórmula para o DCG em uma posição \( p \) é:
   \[
   DCG_p = rel_1 + \sum_{i=2}^{p} \frac{rel_i}{\log_2(i+1)}
   \]
   onde \( rel_i \) é a relevância do item na posição \( i \).

3. **Normalized DCG (NDCG):**  
   O NDCG é obtido dividindo o DCG do ranking gerado pelo DCG do ranking ideal (IDCG), que é o máximo ganho cumulativo possível. Assim, o NDCG varia de 0 a 1, onde 1 indica que o modelo gerou a ordenação ideal.
   \[
   NDCG_p = \frac{DCG_p}{IDCG_p}
   \]

### Interpretação do NDCG@10

- **NDCG@10 = 1:**  
  O modelo gerou a ordenação perfeita para os 10 primeiros itens. Isso significa que as notícias mais relevantes estão posicionadas nas primeiras posições.

- **NDCG@10 próximo de 1:**  
  O modelo tem uma boa performance, com a maioria dos itens relevantes aparecendo no topo da lista. Pequenas discrepâncias podem ocorrer, mas a ordenação é bastante eficiente.

- **NDCG@10 baixo (próximo de 0):**  
  A ordenação gerada pelo modelo está muito distante da ideal. Itens relevantes estão posicionados em posições inferiores, indicando que o modelo precisa ser melhorado.

Em sistemas de recomendação, valores altos de NDCG são desejáveis, pois garantem que os usuários receberão as recomendações mais pertinentes no topo da lista, aumentando a probabilidade de engajamento.

---

## Fluxo do Pipeline de Avaliação

1. **Preparação dos Dados de Avaliação:**  
   O pipeline recebe um DataFrame de avaliação que contém as features (divididas em features de clientes e de notícias) e a coluna `TARGET`, que representa a relevância real do par usuário-notícia.

2. **Construção do Input para Predição:**  
   As features dos clientes e das notícias são extraídas e combinadas horizontalmente para formar um DataFrame com as colunas esperadas pelo modelo (definidas em `EXPECTED_COLUMNS`).

3. **Conversão dos Tipos de Dados:**  
   As colunas são convertidas para os tipos esperados pelo modelo, por exemplo, `isWeekend` para booleano e as demais para valores numéricos (float).

4. **Predição:**  
   O modelo gera os scores para cada par usuário-notícia utilizando o método `predict`.

5. **Cálculo do NDCG@10:**  
   O pipeline calcula o NDCG@10 comparando os scores preditos com os valores reais do `TARGET`. Essa métrica reflete a qualidade da ordenação gerada pelo modelo.

6. **Retorno das Métricas:**  
   O pipeline retorna um dicionário com a métrica calculada, por exemplo, `{"NDCG_10": valor}`.

---

## Como Executar

A execução do pipeline de avaliação é realizada via comando `make evaluate` (ou similar), que executa o script localizado em `src/evaluation/pipeline.py`. Esse script:

- Carrega os dados de avaliação.
- Carrega o modelo treinado (via MLflow ou localmente).
- Executa a função `evaluate_model` para calcular as métricas.
- Exibe as métricas de avaliação, permitindo a análise do desempenho do modelo.

---

## Referências

- [LightGBM Documentation](https://lightgbm.readthedocs.io/)
- [Lambdarank: Optimizing Search Engines Using Listwise Learning to Rank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf)
- [NDCG Score - scikit-learn documentation](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.ndcg_score.html)
- [MLflow Model Registry](https://mlflow.org/docs/latest/model-registry.html)
- [Pandas Documentation](https://pandas.pydata.org/docs/)
- [Pydantic Documentation](https://pydantic-docs.helpmanual.io/)

---
