import os
import time
from typing import List, Optional, Dict, Union
from contextlib import asynccontextmanager

import mlflow
import pandas as pd
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from mlflow.tracking import MlflowClient

from src.predict.pipeline import predict_for_userId
from src.config import get_config, USE_S3, configure_logger
from src.storage.io import Storage
from src.data.data_loader import load_data_for_prediction
from src.recommendation_model.mocked_model import MockedRecommender

# Configura o logger centralizado
logger = configure_logger("api")

# Cache global para dados
DATA_CACHE: Dict[str, pd.DataFrame] = {}

# Função para carregar o modelo via MLflow com medição de tempo


def load_mlflow_model():
    start_time = time.time()
    model_name = get_config("MODEL_NAME", "news-recommender")
    model_alias = get_config("MODEL_ALIAS", "champion")
    try:
        mlflow_tracking_uri = get_config("MLFLOW_TRACKING_URI")
        if mlflow_tracking_uri:
            mlflow.set_tracking_uri(mlflow_tracking_uri)
        model_uri = f"models:/{model_name}@{model_alias}"
        model = mlflow.pyfunc.load_model(model_uri)
        load_time = time.time() - start_time
        logger.info(f"Modelo carregado: {model_name}@{model_alias} em {load_time:.2f} segundos")
        return model
    except Exception as e:
        logger.error(f"Erro ao carregar modelo do MLflow: {e}")
        logger.warning("Usando modelo mockado devido a erro ao carregar do MLflow.")
        return MockedRecommender()


# Função para carregar os dados de predição e armazená-los em cache


def load_prediction_data() -> Dict[str, pd.DataFrame]:
    start_time = time.time()
    if "prediction_data" in DATA_CACHE:
        logger.info("Usando dados em cache para predição.")
        return DATA_CACHE["prediction_data"]

    try:
        logger.info("Carregando dados para predição (primeira vez)...")
        storage = Storage(use_s3=USE_S3)
        # Inclui metadados se disponível
        data = load_data_for_prediction(storage, include_metadata=True)

        # Otimização: Converter colunas numéricas para tipos mais eficientes
        for df_name, df in data.items():
            for col in df.columns:
                if df[col].dtype == "float64":
                    # Downcasting de float64 para float32
                    df[col] = pd.to_numeric(df[col], downcast="float")
                elif df[col].dtype == "int64":
                    # Downcasting de int64 para int32/int16
                    df[col] = pd.to_numeric(df[col], downcast="integer")

        # Otimização: Pré-calcular estatísticas úteis
        if "news_features" in data:
            data["news_count"] = len(data["news_features"])

        load_time = time.time() - start_time
        logger.info(f"Dados carregados e otimizados em {load_time:.2f} segundos.")

        DATA_CACHE["prediction_data"] = data
        return data
    except Exception as e:
        logger.error(f"Erro ao carregar dados para predição: {e}")
        raise e


# Dependências para injeção via FastAPI


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
        if hasattr(model, "metadata") and hasattr(model.metadata, "get"):
            version = model.metadata.get("mlflow.runName")
            if version:
                return version
        # Caso contrário, consulta o MLflow Registry via alias
        model_name = get_config("MODEL_NAME", "news-recommender")
        model_alias = get_config("MODEL_ALIAS", "champion")
        client = MlflowClient()
        model_version = client.get_model_version_by_alias(model_name, model_alias)
        if model_version:
            return model_version.version
        return "unknown"
    except Exception as e:
        logger.error(f"Erro ao obter versão do modelo: {e}")
        return "unknown"


# Modelos Pydantic


class PredictRequest(BaseModel):
    userId: str = Field(
        default="4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297",
        example="4b3c2c5c0edaf59137e164ef6f7d88f94d66d0890d56020de1ca6afd55b4f297",
        description="ID do usuário para recomendação",
    )
    max_results: int = Field(default=5, example=5, description="Número máximo de recomendações")
    min_score: float = Field(
        default=0.3,
        alias="minScore",
        example=0.3,
        description="Score mínimo para considerar uma recomendação",
    )

    class Config:
        populate_by_field_name = True


class NewsItem(BaseModel):
    news_id: str = Field(..., description="ID da notícia")
    score: Union[float, str] = Field(..., description="Score de recomendação")
    title: Optional[str] = Field(None, description="Título da notícia (se disponível)")
    url: Optional[str] = Field(None, description="URL da notícia (se disponível)")
    issuedDate: Optional[str] = Field(None, description="Data de emissão da notícia")
    issuedTime: Optional[str] = Field(None, description="Hora de emissão da notícia")


class PredictResponse(BaseModel):
    userId: str = Field(..., description="ID do usuário")
    recommendations: List[NewsItem] = Field(..., description="Lista de recomendações")
    model_version: str = Field(..., description="Versão do modelo usado")
    cold_start: bool = Field(False, description="Indica se o usuário é cold start")
    processing_time_ms: float = Field(..., description="Tempo de processamento em ms")
    timing_details: Optional[Dict[str, float]] = Field(
        None, description="Detalhes de tempo por etapa"
    )


class HealthResponse(BaseModel):
    status: str = Field(..., description="Status da API")
    model_status: str = Field(..., description="Status do modelo")
    model_version: str = Field(..., description="Versão do modelo")
    environment: str = Field(..., description="Ambiente de execução")
    data_loaded: bool = Field(..., description="Status dos dados")


# Cria a aplicação FastAPI
app = FastAPI(title="Recomendação de Notícias API", version="1.0.0")

# Configuração de CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Função lifespan para inicialização e finalização do app


@asynccontextmanager
async def lifespan(app: FastAPI):
    start_time = time.time()
    logger.info("Iniciando API de Recomendação de Notícias")
    try:
        # Pré-carrega o modelo e os dados durante a inicialização
        app.state.model = load_mlflow_model()
        app.state.prediction_data = load_prediction_data()

        init_time = time.time() - start_time
        logger.info(f"Modelo e dados carregados com sucesso em {init_time:.2f} segundos")
    except Exception as e:
        logger.error(f"Erro na inicialização: {e}")
    yield
    logger.info("Desligando API de Recomendação de Notícias")
    DATA_CACHE.clear()


# Aplica o handler de lifespan à aplicação
app.router.lifespan_context = lifespan

# Endpoints da API


@app.get("/health", response_model=HealthResponse, tags=["Monitoring"])
async def health_check(model=Depends(get_model), data=Depends(get_prediction_data)):
    try:
        model_version = get_model_version(model)
        return HealthResponse(
            status="ok",
            model_status="loaded",
            model_version=model_version,
            environment=os.getenv("ENV", "dev"),
            data_loaded=len(data) > 0,
        )
    except Exception as e:
        logger.error(f"Erro no health check: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# Versão otimizada da rota de predição com métricas de tempo


@app.post("/predict", response_model=PredictResponse, tags=["Prediction"])
def predict(request: PredictRequest):
    start_time = time.time()
    timing = {}  # Dicionário para armazenar métricas de tempo

    try:
        # Timer para obtenção de dependências
        deps_start = time.time()
        model = get_model()
        prediction_data = get_prediction_data()
        news_features_df = prediction_data["news_features"]
        clients_features_df = prediction_data["clients_features"]
        timing["dependencies"] = time.time() - deps_start

        # Timer para a predição
        predict_start = time.time()
        rec_entries, cold_start_flag = predict_for_userId(
            userId=request.userId,
            news_features_df=news_features_df,
            clients_features_df=clients_features_df,
            model=model,
            n=request.max_results,
            score_threshold=request.min_score,
        )
        timing["prediction"] = time.time() - predict_start

        # Timer para formatação da resposta
        format_start = time.time()
        rec_items = []
        for rec in rec_entries:
            news_id = rec.get("pageId")
            try:
                score_value = float(rec.get("score", 0))
                rounded_score = round(score_value, 2)
            except (ValueError, TypeError):
                rounded_score = rec.get("score", 0)
            rec_items.append(
                NewsItem(
                    news_id=news_id,
                    score=rounded_score,
                    title=rec.get("title"),
                    url=rec.get("url"),
                    issuedDate=rec.get("issuedDate"),
                    issuedTime=rec.get("issuedTime"),
                )
            )
        timing["formatting"] = time.time() - format_start

        processing_time_ms = (time.time() - start_time) * 1000
        timing["total_ms"] = processing_time_ms

        # Log de métricas de performance
        logger.info(
            f"""Predição para {request.userId}: {len(rec_items)}
             recomendações em {processing_time_ms:.2f}ms"""
        )
        logger.info(f"Métricas de tempo: {timing}")

        return PredictResponse(
            userId=request.userId,
            recommendations=rec_items,
            model_version=get_model_version(model),
            cold_start=cold_start_flag,
            processing_time_ms=processing_time_ms,
            timing_details=timing,
        )
    except Exception as e:
        error_time = (time.time() - start_time) * 1000
        logger.error(f"Erro na predição após {error_time:.2f}ms: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/info", tags=["Monitoring"])
async def model_info(model=Depends(get_model)):
    try:
        try:
            if hasattr(model, "metadata") and hasattr(model.metadata, "to_dict"):
                metadata = model.metadata.to_dict()
            else:
                metadata = {"warning": "Metadados não disponíveis"}
        except Exception as e:
            logger.warning(f"Metadados não disponíveis: {e}")
            metadata = {"warning": "Metadados não disponíveis"}

        # Adicionando informações sobre o cache de dados
        cache_info = {
            "cache_hit": "prediction_data" in DATA_CACHE,
            "cache_size": len(DATA_CACHE),
        }

        if "prediction_data" in DATA_CACHE:
            cache_info["news_count"] = len(DATA_CACHE["prediction_data"].get("news_features", []))
            cache_info["clients_count"] = len(
                DATA_CACHE["prediction_data"].get("clients_features", [])
            )

        return {
            "model_version": get_model_version(model),
            "environment": os.getenv("ENV", "dev"),
            "metadata": metadata,
            "cache": cache_info,
            "timestamp": pd.Timestamp.now().isoformat(),
        }
    except Exception as e:
        logger.error(f"Erro ao obter informações do modelo: {e}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    host = get_config("API_HOST", "0.0.0.0")
    port = int(get_config("API_PORT", 8000))
    uvicorn.run(app, host=host, port=port, reload=True)
