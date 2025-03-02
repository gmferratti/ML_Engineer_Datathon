import os
import pandas as pd
import numpy as np
import warnings
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from src.config import logger, DATA_PATH
from storage.io import Storage
from typing import Dict, Any, Tuple, List, Optional


def prepare_features(raw_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepara os dados para treino aplicando frequency encoding e divisão em treino/teste.
    """
    logger.info("🔍 [Utils] Separando features do target...")
    TARGET_COLS = ["userId", "pageId", "coldStart", "TARGET"]
    y = raw_data[TARGET_COLS]
    X = raw_data.drop(columns=["TARGET"])

    cold_regs = X[X["coldStart"]].shape[0]
    logger.info("🧹 [Utils] Removendo %d registros de cold start...", cold_regs)
    X = X[~X["coldStart"]]
    y = y[~y["coldStart"]]

    non_cold = X.shape[0]
    prop = round(100 * (cold_regs / (non_cold + cold_regs)), 2)
    logger.info("📈 [Utils] Proporção de cold start: %.2f%%", prop)

    logger.info("🔀 [Utils] Dividindo dados em treino e teste...")
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    cat_cols = [
        col
        for col in X_train.select_dtypes(include=["object", "category"]).columns
        if col not in ("userId", "pageId")
    ]
    encoder_mapping = {}
    logger.info("🔠 [Utils] Aplicando Frequency Encoding nas variáveis categóricas...")
    for col in cat_cols:
        freq = X_train[col].value_counts(normalize=True)
        encoder_mapping[col] = freq.to_dict()
        new_col = f"{col}Freq"
        X_train[new_col] = X_train[col].map(freq)
        X_test[new_col] = X_test[col].map(freq).astype(float).fillna(0)

    logger.info("🗑️ [Utils] Removendo identificadores e colunas redundantes...")
    KEY_COLS = ["userId", "pageId", "issuedDatetime", "timestampHistoryDatetime"]
    URL_COLS = ["localState", "localRegion", "themeMain", "themeSub"]
    REDUNDANT = ["userType", "dayPeriod", "coldStart"]
    DROP_COLS = KEY_COLS + URL_COLS + REDUNDANT

    group_train = X_train.groupby("userId").size().reset_index(name="groupCount")
    group_test = X_test.groupby("userId").size().reset_index(name="groupCount")

    if group_train["groupCount"].sum() != len(X_train):
        warnings.warn("Soma dos grupos diferente em X_train")
    if group_test["groupCount"].sum() != len(X_test):
        warnings.warn("Soma dos grupos diferente em X_test")

    X_train_red = X_train.drop(columns=DROP_COLS, errors="ignore")
    X_test_red = X_test.drop(columns=DROP_COLS, errors="ignore")

    logger.info("✅ [Utils] Dados preparados com sucesso!")
    return {
        "X_train_full": X_train.reset_index(drop=True),
        "X_train": X_train_red.reset_index(drop=True),
        "X_test_full": X_test.reset_index(drop=True),
        "X_test": X_test_red.reset_index(drop=True),
        "y_train": y_train["TARGET"].reset_index(drop=True),
        "y_test": y_test["TARGET"].reset_index(drop=True),
        "encoder_mapping": encoder_mapping,
        "group_train": group_train.reset_index(drop=True),
        "group_test": group_test.reset_index(drop=True),
    }


def load_train_data(storage: Optional[Storage] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados de treino a partir de arquivos Parquet.
    """
    if storage is None:
        from src.config import USE_S3

        storage = Storage(use_s3=USE_S3)
    base_train = os.path.join(DATA_PATH, "train")
    X_path = os.path.join(base_train, "X_train.parquet")
    y_path = os.path.join(base_train, "y_train.parquet")
    logger.info("🔄 [Utils] Carregando X_train de: %s", X_path)
    X_train = storage.read_parquet(X_path)
    logger.info("🔄 [Utils] Carregando y_train de: %s", y_path)
    y_train = storage.read_parquet(y_path)
    logger.info(
        "✅ [Utils] Dados carregados: X_train %s, y_train %s", X_train.shape, y_train.shape
    )
    return X_train, y_train


def feature_selection(
    suggested_feats: pd.DataFrame,
    df_target: pd.DataFrame,
    target_col: str = "TARGET",
    method: str = "correlation",
    drop_cols: Optional[List[str]] = None,
    threshold: float = 0.9,
    k_best: int = 10,
) -> pd.DataFrame:
    """
    Seleciona features com base em correlação ou teste univariado.
    """
    logger.info("🔎 [Utils] Selecionando features usando o método '%s'...", method)
    df = suggested_feats.merge(df_target, on=["userId", "pageId"])
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if method == "correlation":
        corr = X.corr().abs()
        upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
        drop_feats = [col for col in upper.columns if any(upper[col] > threshold)]
        logger.info("🗑️ [Utils] Features removidas por alta correlação: %s", drop_feats)
        result = pd.concat([X.drop(columns=drop_feats), y], axis=1)
    elif method == "univariate":
        selector = SelectKBest(score_func=f_classif, k=k_best)
        selector.fit(X, y)
        mask = selector.get_support()
        sel_feats = X.columns[mask]
        logger.info("✅ [Utils] Features selecionadas (univariate): %s", list(sel_feats))
        result = pd.concat([X[sel_feats], y], axis=1)
    else:
        raise ValueError("Método desconhecido. Escolha 'correlation' ou 'univariate'.")

    logger.info("🔚 [Utils] Seleção de features concluída: Resultado shape: %s", result.shape)
    return result
