# LightGBMRanker - Modelo de Aprendizado de Ranking

Este documento descreve o funcionamento do `LightGBMRanker`, que é uma implementação de um modelo de **aprendizado de ranking** baseado no [LightGBM](https://lightgbm.readthedocs.io/), utilizando como objetivo de otimização o [lambdarank](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/MSR-TR-2010-82.pdf). O LightGBM (Light Gradient Boosting Machine) é uma biblioteca de machine learning desenvolvida pela Microsoft que implementa técnicas de boosting de gradiente, ou seja, métodos que combinam várias árvores de decisão de forma sequencial para melhorar a performance preditiva.

---

## Sumário
1. Visão Geral
2. Estrutura de Arquivos
3. Classe `BaseRecommender`
4. Classe `LightGBMRanker`
   - Inicialização
   - Treinamento (`train`)
   - Predição (`predict`)
5. Fluxo de Execução
6. Parâmetros do Modelo

---

## Visão Geral

O **LightGBMRanker** é um modelo especializado em problemas de ranking. Diferentemente de um modelo de predição tradicional (regressão ou classificação), um modelo de ranking se preocupa em **ordenar** itens para um determinado contexto (ex. usuário, sessão, etc.). Fornecemos duas classes principais em `base_model.py`:

- **`BaseRecommender`** (classe abstrata): Define a estrutura mínima para um modelo de recomendação/ranking.
- **`LightGBMRanker`**: Especializa a classe `BaseRecommender` utilizando o **LightGBM** com o objetivo `lambdarank`.

Além delas, há uma outra classe chamada **`MockedRecommender`**, cujo principal intuito foi mockar algo rápido para testarmos as integrações com a API e MLFlow. Neste momento, podemos ignorá-la.

---

## Classe `BaseRecommender`

A classe `BaseRecommender` é uma classe abstrata que define os métodos básicos a serem implementados por qualquer recomendador:

```python
from abc import ABC, abstractmethod

class BaseRecommender(ABC):
    def __init__(self, params=None, num_boost_round=100):
        self.params = params if params is not None else {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'learning_rate': 0.05,
            'num_leaves': 31,
            'verbose': -1
        }
        self.num_boost_round = num_boost_round
        self.model = None

    @abstractmethod
    def predict(self, model_input):
        pass

    @abstractmethod
    def train(self, X, y):
        pass
```
## Principais Componentes

Na classe `BaseRecommender`:

- **`__init__`**:  
  - **`params`** (dict): Parâmetros específicos do LightGBM. Se não fornecidos, utiliza um conjunto padrão voltado para ranking.  
  - **`num_boost_round`** (int): Número de árvores (iterações) usadas no treinamento.  
  - **`model`**: Inicialmente `None`; será atualizado após a etapa de treinamento.

- **`train(X, y)`**:  
  Método abstrato para treinar o modelo. A implementação concreta deve considerar também informações de grupo (ex.: grupo de usuários).

- **`predict(model_input)`**:  
  Método abstrato para gerar predições. Espera que `model_input` seja um dicionário contendo, por exemplo, as chaves `"client_features"` e `"news_features"`.

---

## Classe LightGBMRanker

Esta classe implementa um modelo de ranking utilizando LightGBM com o objetivo `lambdarank`, herdando da `BaseRecommender`.

### Inicialização

```python
import lightgbm as lgb

class LightGBMRanker(BaseRecommender):
    def __init__(self, params=None, num_boost_round=100):
        super().__init__(params=params, num_boost_round=num_boost_round)
```

## Treinamento (`train`)

Nesta seção, o método `train` treina o modelo LightGBM em modo de ranking (LambdaRank).  
Ele recebe como entrada:
- **X**: A matriz de features com dimensões `[n_amostras x n_features]`.
- **y**: O vetor de relevâncias (scores) para cada amostra.
- **group**: Um vetor ou lista que indica a quantidade de amostras por grupo (por exemplo, interações por usuário). A soma dos valores deve corresponder ao total de linhas em **X**.

```python
def train(self, X, y, group):
    """
    Treina o modelo LightGBM em modo de ranking (LambdaRank).

    Args:
        X (array-like): Matriz de features [n_amostras x n_features].
        y (array-like): Vetor de relevâncias (scores) para cada amostra.
        group (list ou array): Quantidade de amostras por grupo (ex.: interações por usuário).
                               A soma dos valores deve ser igual ao número total de linhas em X.
    """
    train_data = lgb.Dataset(X, label=y, group=group)
    self.model = lgb.train(
        self.params,
        train_data,
        num_boost_round=self.num_boost_round
    )
```

Após criar um objeto `lgb.Dataset` com as features, os labels e o agrupamento, o modelo é treinado com os parâmetros definidos e o número de iterações especificado.

---

## Predição (`predict`)

O método `predict` gera os scores de relevância para os itens, utilizando as features dos clientes e dos itens.  
Ele espera que o dicionário de entrada (`model_input`) contenha as chaves:
- **client_features**: Dados referentes ao usuário (por exemplo, comportamento ou preferências).
- **news_features**: Dados referentes ao item (como notícia ou produto).

```python
import numpy as np

def predict(self, model_input):
    """
    Realiza a predição (score) para ranqueamento.

    Args:
        model_input (dict): Deve conter:
            - 'client_features': Dados referentes ao usuário.
            - 'news_features': Dados referentes ao item.
    
    Returns:
        np.ndarray: Array com os scores preditos, para ordenação dos itens.
    """
    client_features = model_input.get('client_features')
    news_features = model_input.get('news_features')

    if client_features is None or news_features is None:
        raise ValueError(
            "O dicionário 'model_input' deve conter as chaves 'client_features' e 'news_features'."
        )

    X = np.concatenate([client_features, news_features], axis=1)
    
    if self.model is None:
        raise ValueError("O modelo ainda não foi treinado. Execute train() antes de predict().")
    
    scores = self.model.predict(X)
    return scores
```

O método concatena as features, verifica se o modelo foi treinado e, então, utiliza o método `predict` do LightGBM para retornar um array de scores. Esses scores são usados para ordenar os itens conforme sua relevância.

---

## Fluxo de Execução

O fluxo de execução a seguir exemplifica como utilizar o modelo:

```python
# Exemplo de uso:

# 1. Instanciação do Modelo
ranker = LightGBMRanker(params=..., num_boost_round=...)

# 2. Preparação dos Dados de Treinamento
#    - X: matriz com shape [n_amostras, n_features]
#    - y: vetor de relevâncias
#    - group: lista/array com o número de amostras por grupo 
#             (ex.: número de notícias consumidas por cada usuário)

# 3. Treinamento do Modelo
ranker.train(X, y, group)

# 4. Realização de Predições
scores = ranker.predict({
    "client_features": X_client, 
    "news_features": X_news
})

# 5. Definição do Ranking
#    (ex.: utilizando argsort para ordenar os scores em ordem decrescente)
```

Cada etapa garante que o modelo seja corretamente instanciado, treinado e utilizado para gerar predições com base nas features fornecidas.

---

## Parâmetros do Modelo

Os principais parâmetros que podem ser customizados ao instanciar o modelo são:

| Parâmetro       | Descrição                                                           | Valor Padrão |
|-----------------|---------------------------------------------------------------------|--------------|
| `objective`     | Define a tarefa de aprendizado. Para ranking, usa `lambdarank`.      | `lambdarank` |
| `metric`        | Métrica para avaliação durante o treinamento. Usualmente `ndcg`.     | `ndcg`       |
| `learning_rate` | Taxa de aprendizado do modelo.                                      | `0.05`       |
| `num_leaves`    | Controla a complexidade das árvores.                                | `31`         |
| `verbose`       | Nível de logs do LightGBM.                                          | `-1`         |

Exemplo de customização:

```python
params = {
    "objective": "lambdarank",
    "metric": "ndcg",
    "learning_rate": 0.1,
    "num_leaves": 63,
    "verbose": -1
}
ranker = LightGBMRanker(params=params, num_boost_round=200)
```

Esta customização permite ajustar o modelo para melhorar seu desempenho em tarefas de ranking.

---
Sinta-se à vontade para abrir **issues** ou **pull requests** com dúvidas e contribuições.