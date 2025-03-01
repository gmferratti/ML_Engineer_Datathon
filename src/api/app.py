import os
import time
import logging
from typing import List, Optional, Dict

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from predict.predict import predict_for_userId
from config import get_config, USE_S3
from storage.io import Storage
from data.data_loader import load_data_for_prediction, load_model

# Configuração de logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("api")

# Modelos Pydantic


class PredictRequest(BaseModel):
    user_id: str = Field(..., description="ID do usuário para recomendação")
    max_results: int = Field(5, description="Número máximo de recomendações")
    min_score: float = Field(0.3, alias="minScore",
                             description="Score mínimo para considerar uma recomendação")

    class Config:
        allow_population_by_field_name = True


class NewsItem(BaseModel):
    news_id: str = Field(..., description="ID da notícia")
    score: float = Field(..., description="Score de recomendação")
    title: Optional[str] = Field(None, description="Título da notícia (se disponível)")
    url: Optional[str] = Field(None, description="URL da notícia (se disponível)")


class PredictResponse(BaseModel):
    user_id: str = Field(..., description="ID do usuário")
    recommendations: List[NewsItem] = Field(..., description="Lista de recomendações")
    model_version: str = Field(..., description="Versão do modelo usado")
    cold_start: bool = Field(False, description="Indica se o usuário é cold start")
    processing_time_ms: float = Field(..., description="Tempo de processamento em ms")


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status da API")
    model_status: str = Field(..., description="Status do modelo")
    model_version: str = Field(..., description="Versão do modelo")
    environment: str = Field(..., description="Ambiente de execução")


# Criação da aplicação FastAPI
app = FastAPI(title="Recomendação de Notícias API", version="1.0.0")

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Inicialização do Storage
storage = Storage(use_s3=USE_S3)

# Caches para dados comuns
DATA_CACHE: Dict[str, pd.DataFrame] = {}


def load_mlflow_model():
    model_name = get_config("MODEL_NAME", "news-recommender")
    model_alias = get_config("MODEL_ALIAS", "champion")
    try:
        mlflow_tracking_uri = get_config("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        logger.info(f"Modelo carregado: {model_name}@{model_alias}")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo do MLflow: {e}")
        # Se falhar ao carregar do MLflow, tenta carregar o modelo salvo localmente/S3
        try:
            return load_model(storage)
        except Exception as e2:
            logger.error(f"Também falhou ao carregar modelo do armazenamento: {e2}")
            from recomendation_model.mocked_model import MockedRecommender
            logger.warning(
                "Usando modelo mockado devido a erro ao carregar do MLflow e do armazenamento")
            return MockedRecommender()


def load_prediction_data() -> Dict[str, pd.DataFrame]:
    if "prediction_data" in DATA_CACHE:
        return DATA_CACHE["prediction_data"]
    try:
        data = load_data_for_prediction(storage)
        DATA_CACHE["prediction_data"] = data
        return data
    except Exception as e:
        logger.error(f"Erro ao carregar dados para predição: {e}")
        # Retorna dados mockados em caso de erro
        news_data = pd.DataFrame({
            "pageId": [f"news_{i}" for i in range(1, 101)],
            # Colunas extras para a resposta
            "title": [f"Notícia {i}" for i in range(1, 101)],
            "url": [f"https://g1.globo.com/noticia/{i}" for i in range(1, 101)]
        })

        user_data = pd.DataFrame({
            "userId": [f"user_{i}" for i in range(1, 21)],
        })

        return {
            "news_features": news_data,
            "clients_features": user_data
        }


def get_model():
    if not hasattr(app.state, "model"):
        app.state.model = load_mlflow_model()
    return app.state.model


def get_prediction_data():
    if not hasattr(app.state, "prediction_data"):
        app.state.prediction_data = load_prediction_data()
    return app.state.prediction_data


def get_model_version(model=Depends(get_model)) -> str:
    try:
        # Tenta obter a versão do modelo MLflow
        if hasattr(model, 'metadata') and hasattr(model.metadata, 'get'):
            return model.metadata.get("mlflow.runName", "unknown")
        # Para modelo carregado do arquivo pickle
        elif hasattr(model, '__version__'):
            return getattr(model, '__version__')
        # Versão padrão
        return "unknown"
    except Exception as e:
        logger.error(f"Erro ao obter versão do modelo: {e}")
        return "unknown"


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check(model=Depends(get_model)):
    try:
        model_version = get_model_version(model)
        return HealthResponse(
            status="ok",
            model_status="loaded",
            model_version=model_version,
            environment=os.getenv("ENV", "dev")
        )
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    start_time = time.time()
    try:
        model = app.state.model
        prediction_data = get_prediction_data()
        news_features_df = prediction_data["news_features"]
        clients_features_df = prediction_data["clients_features"]

        # Obtém uma lista de IDs das notícias recomendadas
        rec_ids = predict_for_userId(
            userId=request.user_id,
            news_features_df=news_features_df,
            clients_features_df=clients_features_df,
            model=model,
            n=request.max_results,
            score_threshold=request.min_score
        )

        # Transforma cada ID em um objeto NewsItem
        rec_items = []
        for news_id in rec_ids:
            # Busca a notícia pelo ID
            row = news_features_df[news_features_df["pageId"] == news_id]
            if not row.empty:
                title = row.iloc[0].get("title") if "title" in row.iloc[0] else None
                url = row.iloc[0].get("url") if "url" in row.iloc[0] else None
            else:
                title = None
                url = None
            # Aqui usamos um score fixo (0.5) como exemplo; ajuste conforme necessário
            rec_items.append(NewsItem(news_id=news_id, score=0.5, title=title, url=url))

        processing_time_ms = (time.time() - start_time) * 1000
        return PredictResponse(
            user_id=request.user_id,
            recommendations=rec_items,
            model_version=get_model_version(model),
            cold_start=False,
            processing_time_ms=processing_time_ms
        )
    except Exception as e:
        logger.error(f"Erro na predição: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", tags=["Monitoring"])
async def model_info(model=Depends(get_model)):
    try:
        # Tenta obter os metadados do modelo
        try:
            if hasattr(model, 'metadata') and hasattr(model.metadata, 'to_dict'):
                metadata = model.metadata.to_dict()
            else:
                metadata = {"warning": "Metadados não disponíveis"}
        except Exception as e:
            logger.warning(f"Metadados não disponíveis: {e}")
            metadata = {"warning": "Metadados não disponíveis"}

        return {
            "model_version": get_model_version(model),
            "environment": os.getenv("ENV", "dev"),
            "metadata": metadata,
            "timestamp": pd.Timestamp.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.on_event("startup")
async def startup_event():
    logger.info("Iniciando API de Recomendação de Notícias")
    try:
        app.state.model = load_mlflow_model()
        app.state.prediction_data = load_prediction_data()
        logger.info("Modelo e dados carregados com sucesso")
    except Exception as e:
        logger.error(f"Erro na inicialização: {e}")


@app.on_event("shutdown")
async def shutdown_event():
    logger.info("Desligando API de Recomendação de Notícias")
    DATA_CACHE.clear()


if __name__ == "__main__":
    import uvicorn
    host = get_config("API_HOST", "0.0.0.0")
    port = int(get_config("API_PORT", 8000))
    uvicorn.run("app:app", host=host, port=port, reload=True)
