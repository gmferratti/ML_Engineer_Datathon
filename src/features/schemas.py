import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema

def get_model_signature():
    """
    Retorna o ModelSignature para as features utilizadas no modelo de engajamento.
    
    As colunas de entrada esperadas são:
      - isWeekend (boolean): é ou não FDS.
      - relLocalState (double): frequência relativa do Estado.
      - relLocalRegion (double): frequência relativa da Região.
      - relThemeMain (double): frequência relativa do tema principal.
      - relThemeSub (double): frequência relativa do subtema.
      - userTypeFreq (double): usuário logado ou não (pós frequency-encoder)
      - dayPeriodFreq (double): período do dia (pós frequency-encoder)
      - localStateFreq (double): informação de estado pós frequency encoder
      - localRegionFreq (double): informação de região pós frequency encoder
      - themeMainFreq (double): informação de tema pós frequency encoder
      - themeSubFreq (double): informação de subtema prós frequency encoder
    
    A saída do modelo é:
      - TARGET (double): Score de engajamento.
    """
    input_schema = Schema([
        ColSpec(DataType.boolean, "isWeekend"),
        ColSpec(DataType.double, "relLocalState"),
        ColSpec(DataType.double, "relLocalRegion"),
        ColSpec(DataType.double, "relThemeMain"),
        ColSpec(DataType.double, "relThemeSub"),
        ColSpec(DataType.double, "userTypeFreq"),
        ColSpec(DataType.double, "dayPeriodFreq"),
        ColSpec(DataType.double, "localStateFreq"),
        ColSpec(DataType.double, "localRegionFreq"),
        ColSpec(DataType.double, "themeMainFreq"),
        ColSpec(DataType.double, "themeSubFreq")
    ])
    output_schema = Schema([ColSpec(DataType.double, "TARGET")])
    return ModelSignature(inputs=input_schema, outputs=output_schema)



def create_mock_input_example():
    """
    Cria um exemplo de input para o modelo.

    Returns:
        pd.DataFrame: Exemplo com features fictícias.
    """
    return pd.DataFrame(
        [
            {
                "userId": "user_123",
                "pageId": "news_456",
                "user_feat1": 0.5,
                "user_feat2": 0.3,
                "news_feat1": 0.8,
                "news_feat2": 0.4,
            }
        ]
    )

def create_valid_input_example():
    """
    Retorna um exemplo válido de input para o modelo,
    contendo as colunas definidas na assinatura.
    """
    return pd.DataFrame({
        "isWeekend": [False],
        "relLocalState": [0.215645],
        "relLocalRegion": [0.164302],
        "relThemeMain": [0.109407],
        "relThemeSub": [0.081684],
        "userTypeFreq": [0.501069],
        "dayPeriodFreq": [0.301338],
        "localStateFreq": [0.146567],
        "localRegionFreq": [0.081901],
        "themeMainFreq": [0.117847],
        "themeSubFreq": [0.125519]
    })