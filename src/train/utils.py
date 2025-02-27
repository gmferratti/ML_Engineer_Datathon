import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from config import logger
from typing import Dict, Any, Tuple, List, Optional


def prepare_features(raw_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepara os dados para treino, aplicando frequency encoding em variáveis
    categóricas (criando novas colunas com o sufixo '_freq') e mantendo os
    valores originais.

    Passos:
      1. Separar target e features (mantendo os identificadores para o merge).
      2. Dividir os dados em treino e teste.
      3. Para cada variável categórica (exceto os identificadores),
         criar uma nova coluna com o frequency encoding e armazenar o
         dicionário de mapeamento em um dicionário.
      4. Remover os identificadores ('userId' e 'pageId') dos conjuntos de
         treino e teste.

    Parâmetros:
        raw_data (DataFrame): Dados brutos contendo features e target.

    Retorna:
        dict: Dicionário contendo X_train, X_test, y_train, y_test e
              encoder_mapping.
    """
    logger.info("Separando features do target...")
    # Separar target e features mantendo os identificadores para o merge
    target_cols = ["userId", "pageId", "TARGET"]
    y = raw_data[target_cols]
    X = raw_data.drop(columns=["TARGET"])

    logger.info("Dividindo dados em treino e teste...")
    # Dividir os dados (ex.: 70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42
    )

    # Lista de colunas categóricas, excluindo os identificadores
    cat_cols: List[str] = [
        col
        for col in X_train.select_dtypes(include=["object", "category"]).columns
        if col not in ("userId", "pageId")
    ]

    encoder_mapping: Dict[str, Dict[Any, float]] = {}

    logger.info(
        "Aplicando Frequency Encoding (criando colunas adicionais) nas variáveis categóricas..."
    )
    # Para cada coluna categórica, calcula a frequência relativa e cria nova coluna
    for col in cat_cols:
        freq_encoding = X_train[col].value_counts(normalize=True)
        encoder_mapping[col] = freq_encoding.to_dict()
        new_col = f"{col}_freq"
        X_train[new_col] = X_train[col].map(freq_encoding)
        X_test[new_col] = (
            X_test[col].map(freq_encoding).astype(float).fillna(0)
        )

    logger.info("Removendo identificadores...")
    # Remover os identificadores que não serão utilizados como features
    KEY_TRAIN_COLS = ["userId", "pageId", 'localState', 'localRegion','themeMain', 'themeSub']
    X_train_reduced = X_train.drop(columns=KEY_TRAIN_COLS, errors="ignore")
    X_test_reduced = X_test.drop(columns=KEY_TRAIN_COLS, errors="ignore")

    trusted_data = {
        "X_train": X_train_reduced,
        "X_test": X_test_reduced,
        "y_train": y_train["TARGET"],
        "y_test": y_test["TARGET"],
        "encoder_mapping": encoder_mapping,
    }

    return trusted_data


def load_train_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados de treino a partir de arquivos Parquet.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames contendo X_train e y_train.
    """
    base_train_path = os.path.join("data", "train")
    X_train_path = os.path.join(base_train_path, "X_train.parquet")
    y_train_path = os.path.join(base_train_path, "y_train.parquet")
    
    logger.info(f"Carregando X_train de {X_train_path}...")
    X_train = pd.read_parquet(X_train_path)
    
    logger.info(f"Carregando y_train de {y_train_path}...")
    y_train = pd.read_parquet(y_train_path)
    
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
    Realiza a seleção de features no DataFrame `suggested_feats`.

    Parâmetros:
        suggested_feats (DataFrame): DataFrame com as features finais.
        df_target (DataFrame): DataFrame contendo o target.
        target_col (str): Coluna alvo (variável dependente).
        method (str): Método de seleção. Pode ser:
                      - "correlation": remove features altamente correlacionadas.
                      - "univariate": seleciona as k melhores features via teste ANOVA.
        drop_cols (list, opcional): Lista de colunas a serem descartadas antes da
                                    seleção (ex.: identificadores).
        threshold (float): Limite de correlação para remoção de features redundantes.
        k_best (int): Número de melhores features a serem selecionadas no método
                      univariado.

    Retorna:
        DataFrame: DataFrame com as features selecionadas e a coluna target.
    """
    # Merge para garantir que as features e o target estejam juntos
    df = suggested_feats.merge(df_target, on=["userId", "pageId"])

    # Remove colunas indesejadas, se especificado
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)

    # Separa as variáveis preditoras (X) e a variável alvo (y)
    X = df.drop(columns=[target_col])
    y = df[target_col]

    if method == "correlation":
        # Calcula a matriz de correlação entre as features
        corr_matrix = X.corr().abs()
        # Considera apenas a parte superior da matriz para evitar redundância
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Identifica features com alta correlação
        features_to_drop = [
            col for col in upper_tri.columns if any(upper_tri[col] > threshold)
        ]
        X_selected = X.drop(columns=features_to_drop)
        logger.info(f"Features removidas por alta correlação: {features_to_drop}")
        return pd.concat([X_selected, y], axis=1)

    elif method == "univariate":
        # Seleção univariada utilizando o teste F (ANOVA)
        selector = SelectKBest(score_func=f_classif, k=k_best)
        selector.fit(X, y)
        mask = selector.get_support()  # Máscara booleana das features selecionadas
        selected_features = X.columns[mask]
        logger.info(f"Features selecionadas pelo método univariado: {list(selected_features)}")
        X_selected = X[selected_features]
        return pd.concat([X_selected, y], axis=1)

    else:
        raise ValueError(
            "Método de seleção desconhecido. Escolha 'correlation' ou 'univariate'."
        )
