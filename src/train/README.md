# Módulo de Treinamento

Este diretório contém os componentes para treinamento e gerenciamento de modelos de recomendação.

## Estrutura do Módulo

- **`core.py`**: Funções centrais reutilizáveis para interação com MLflow.
- **`pipeline.py`**: Pipeline completo de treinamento com preparação de dados e avaliação.
- **`train.py`**: Script simplificado para experimentos rápidos de treinamento.
- **`utils.py`**: Funções auxiliares para preparação de dados e carregamento.

## Uso

### Pipeline Completo

O pipeline completo é ideal para treinamento em produção:

```python
from train import train_pipeline

# Executa o pipeline completo:
# 1. Carrega e prepara dados
# 2. Treina o modelo LightGBM
# 3. Avalia e registra no MLflow
train_pipeline()
```

### Treinamento Simplificado

Para experimentos rápidos, use o script simplificado:

```python
from train import train_simple
from recomendation_model.base_model import LightGBMRanker

# Treina um modelo mockado (padrão)
model = train_simple(model_params={"threshold": 0.5})

# Ou especifique a classe do modelo
custom_model = train_simple(
    model_class=LightGBMRanker,
    model_params={"learning_rate": 0.05},
    model_name="custom-ranker"
)
```

## Integração com MLflow

Todos os modelos treinados são automaticamente:

1. Registrados no MLflow com parâmetros e métricas
2. Adicionados ao Model Registry com versão
3. Marcados com alias "champion" (a versão mais recente)

## Funções Utilitárias MLflow

O módulo `core.py` fornece funções reutilizáveis para MLflow:

```python
from train.core import log_model_to_mlflow, load_model_from_mlflow, log_basic_metrics

# Registra um modelo no MLflow
log_model_to_mlflow(model, model_name="custom-model")

# Registra métricas básicas (e opcionalmente adicionais)
log_basic_metrics(X_train, metrics={"precision": 0.92, "recall": 0.87})

# Carrega um modelo do MLflow
loaded_model = load_model_from_mlflow(model_name="custom-model", model_alias="production")

# Gera um nome de execução baseado em timestamp
run_name = get_run_name(model_name="custom-model")
```

## Convenções de Nomenclatura

Os experimentos MLflow seguem a convenção de nomenclatura `{model_name}-{YYYYmmdd-HHMMSS}`, permitindo:

- Identificação clara do tipo de modelo
- Fácil ordenação cronológica 
- Rastreabilidade para diagnóstico de problemas