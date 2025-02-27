import pandas as pd
from mlflow.models.signature import ModelSignature
from mlflow.types.schema import ColSpec, DataType, Schema


def get_model_signature():
    """
    Retorna o ModelSignature para as features utilizadas no modelo de engajamento.

    As colunas de entrada esperadas são:
      - userId (string): Identificador único do usuário.
      - pageId (string): Identificador único da notícia.
      - userType (string): Tipo do usuário (ex.: "logged_in" ou "guest").
      - isWeekend (boolean): Flag que indica se o consumo ocorreu no fim de semana.
      - dayPeriod (string): Período do dia em que a notícia foi consumida (ex.: "morning", "afternoon", "night").
      - issuedDatetime (string): Data/hora de publicação da notícia.
      - timestampHistoryDatetime (string): Data/hora de consumo da notícia.
      - coldStart (boolean): Indica se o usuário é novo na plataforma (True para menos de 5 notícias).
      - localState (string): Estado extraído da URL.
      - localRegion (string): Microrregião extraída da URL.
      - themeMain (string): Tema principal da notícia.
      - themeSub (string): Subtema da notícia.
      - relLocalState (double): Fração de consumo do usuário para o estado.
      - relLocalRegion (double): Fração de consumo do usuário para a microrregião.
      - relThemeMain (double): Fração de consumo do usuário para o tema principal.
      - relThemeSub (double): Fração de consumo do usuário para o subtema.

    A saída do modelo é:
      - TARGET (double): Score de engajamento.
    """
    return ModelSignature(
        inputs=Schema(
            [
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
                ColSpec(DataType.double, "relThemeSub"),
            ]
        ),
        outputs=Schema([ColSpec(DataType.double, "TARGET")]),
    )


def create_mock_input_example():
    """
    Exemplo simples com estrutura usada em produção.
    
    Este exemplo mantém o formato simplificado original, com features fictícias:
      - userId, pageId, user_feat1, user_feat2, news_feat1, news_feat2.
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
