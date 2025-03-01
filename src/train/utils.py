import pandas as pd
import numpy as np
import os
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from config import logger, DATA_PATH
from storage.io import Storage
from typing import Dict, Any, Tuple, List, Optional


def prepare_features(raw_data: pd.DataFrame) -> Dict[str, Any]:
    """
    Prepara os dados para treino, aplicando frequency encoding em variáveis
    categóricas (criando novas colunas com o sufixo 'Freq') e mantendo os
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
    TARGET_COLS = ["userId", "pageId", "coldStart", "TARGET"]

    y = raw_data[TARGET_COLS]
    X = raw_data.drop(columns=["TARGET"])

    # Selecionando somente clientes que não são cold_start
    cold_start_regs = (X[X["coldStart"]]).shape[0]

    logger.info(f"Removido {cold_start_regs} registros cold start")

    X = X[~X["coldStart"]]
    y = y[~y["coldStart"]]

    non_cold_start_regs = X.shape[0]
    cold_start_proportion = round(
        100 * (cold_start_regs/(non_cold_start_regs+cold_start_regs)), 2)

    logger.info(f"Proporção de registros cold_start: {cold_start_proportion} %")

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
        new_col = f"{col}Freq"
        X_train[new_col] = X_train[col].map(freq_encoding)
        X_test[new_col] = (
            X_test[col].map(freq_encoding).astype(float).fillna(0)
        )

    logger.info("Removendo identificadores...")

    # Remover os identificadores que não serão utilizados como features
    KEY_TRAIN_COLS = [
        'userId',
        'pageId',
        'issuedDatetime',
        'timestampHistoryDatetime',
    ]
    URL_COLS = [
        'localState',
        'localRegion',
        'themeMain',
        'themeSub',
    ]
    REDUNDANT_UNNECESSARY = ['userType', 'dayPeriod', 'coldStart']
    COLS_TO_DROP = KEY_TRAIN_COLS + URL_COLS + REDUNDANT_UNNECESSARY

    # Cria DataFrame com o número de interações por usuário (groupCount)
    group_train = X_train.groupby("userId").size().reset_index(name="groupCount")
    group_test = X_test.groupby("userId").size().reset_index(name="groupCount")

    # Verifica se a soma dos grupos é igual ao número total de linhas do dataset
    assert group_train["groupCount"].sum() == len(
        X_train), "A soma dos grupos deve ser igual ao número total de linhas em X_train"
    assert group_test["groupCount"].sum() == len(
        X_test), "A soma dos grupos deve ser igual ao número total de linhas em X_test"

    X_train_reduced = X_train.drop(columns=COLS_TO_DROP, errors="ignore")
    X_test_reduced = X_test.drop(columns=COLS_TO_DROP, errors="ignore")

    trusted_data = {
        "X_train_full": X_train.reset_index(drop=True),
        "X_train": X_train_reduced.reset_index(drop=True),
        "X_test_full": X_test.reset_index(drop=True),
        "X_test": X_test_reduced.reset_index(drop=True),
        "y_train": y_train["TARGET"].reset_index(drop=True),
        "y_test": y_test["TARGET"].reset_index(drop=True),
        "encoder_mapping": encoder_mapping,
        "group_train": group_train.reset_index(drop=True),
        "group_test": group_test.reset_index(drop=True)
    }

    return trusted_data


def load_train_data(storage: Optional[Storage] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Carrega os dados de treino a partir de arquivos Parquet.

    Args:
        storage (Storage, opcional): Instância de Storage para I/O.
            Se não for fornecido, cria uma nova instância.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: DataFrames contendo X_train e y_train.
    """
    # Se não for fornecido um objeto Storage, cria um novo
    if storage is None:
        from config import USE_S3
        storage = Storage(use_s3=USE_S3)

    # Define os caminhos dos arquivos
    base_train_path = os.path.join(DATA_PATH, "train")
    X_train_path = os.path.join(base_train_path, "X_train.parquet")
    y_train_path = os.path.join(base_train_path, "y_train.parquet")

    logger.info(f"Carregando X_train de {X_train_path}...")
    X_train = storage.read_parquet(X_train_path)

    logger.info(f"Carregando y_train de {y_train_path}...")
    y_train = storage.read_parquet(y_train_path)

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
