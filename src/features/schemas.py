import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema


def get_model_signature():
    """
    Retorna a assinatura do modelo de engajamento.

    Returns:
        ModelSignature: Assinatura com inputs e output.
    """
    inputs = Schema([
        ColSpec(DataType.string, "userId"),
        ColSpec(DataType.string, "pageId"),
        ColSpec(DataType.string, "userType"),
        ColSpec(DataType.boolean, "isWeekend"),
        ColSpec(DataType.string, "dayPeriod"),
        ColSpec(DataType.string, "issuedDatetime"),
        ColSpec(DataType.string, "timestampHistoryDatetime"),
        ColSpec(DataType.boolean, "coldStart"),
        ColSpec(DataType.string, "localState"),
        ColSpec(DataType.string, "localRegion"),
        ColSpec(DataType.string, "themeMain"),
        ColSpec(DataType.string, "themeSub"),
        ColSpec(DataType.double, "relLocalState"),
        ColSpec(DataType.double, "relLocalRegion"),
        ColSpec(DataType.double, "relThemeMain"),
        ColSpec(DataType.double, "relThemeSub")
    ])
    outputs = Schema([ColSpec(DataType.double, "TARGET")])
    return ModelSignature(inputs=inputs, outputs=outputs)


def create_mock_input_example():
    """
    Cria um exemplo de input para o modelo.

    Returns:
        pd.DataFrame: Exemplo com features fictícias.
    """
    return pd.DataFrame([{
        "userId": "user_123",
        "pageId": "news_456",
        "user_feat1": 0.5,
        "user_feat2": 0.3,
        "news_feat1": 0.8,
        "news_feat2": 0.4,
    }])
