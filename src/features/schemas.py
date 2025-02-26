import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema


def get_model_signature():
    """Schema dinâmico para features combinadas"""
    return ModelSignature(
        #TODO: atualizar aqui com as features finalizadas
        inputs=Schema(
            [
                ColSpec(DataType.string, "userId"),
                ColSpec(DataType.string, "pageId"),
                ColSpec(DataType.double, "user_feat1"),
                ColSpec(DataType.double, "user_feat2"),
                ColSpec(DataType.double, "news_feat1"),
                ColSpec(DataType.double, "news_feat2"),
            ]
        ),
        outputs=Schema([ColSpec(DataType.double)]),
    )


def create_mock_input_example():
    """Exemplo com estrutura real usada em produção"""
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
