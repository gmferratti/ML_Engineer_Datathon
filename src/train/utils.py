import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from lightgbm import LGBMRegressor
from config import logger

def prepare_features(raw_data):
    """
    Prepara os dados para treino sem a seleção de features por alta correlação, 
    utilizando frequency encoding para variáveis categóricas, mas mantendo os valores originais.
    
    Passos:
      1. Separar target e features (mantendo os identificadores para o merge).
      2. Dividir os dados em treino e teste.
      3. Para cada variável categórica (exceto os identificadores), 
         criar uma nova coluna com o frequency encoding (com sufixo '_freq'),
         mantendo a coluna original inalterada.
         Armazenar o dicionário de mapeamento (de-para) em um dicionário.
      4. Remover os identificadores ('userId' e 'pageId') dos conjuntos de treino e teste.
    """
    logger.info("Separando features do target...")
    # Separar target e features mantendo os identificadores para o merge
    y = raw_data[['userId', 'pageId', 'TARGET']]
    X = raw_data.drop(columns=['TARGET'])
    
    logger.info("Dividindo dados em treino e teste...")
    # Dividir os dados em treino e teste (ex.: 70% treino, 30% teste)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    
    # Lista de colunas categóricas, excluindo os identificadores
    cat_cols = [col for col in X_train.select_dtypes(include=['object', 'category']).columns 
                if col not in ['userId', 'pageId']]
    
    # Dicionário para armazenar os mapeamentos de cada coluna
    encoder_mapping = {}
    
    logger.info("Aplicando Frequency Encoding (criando colunas adicionais) nas variáveis categóricas...")
    # Para cada coluna categórica, calcula a frequência e cria nova coluna com o encoded
    for col in cat_cols:
        # Calcula a frequência relativa para cada categoria no conjunto de treino
        freq_encoding = X_train[col].value_counts(normalize=True)
        # Armazena o mapeamento original (de texto para frequência) para a coluna
        encoder_mapping[col] = freq_encoding.to_dict()
        # Cria a coluna com os valores codificados para treino
        X_train[col] = X_train[col].map(freq_encoding)
        # Para o conjunto de teste, utiliza o mesmo mapeamento; valores desconhecidos são preenchidos com 0
        X_test[col] = X_test[col].map(freq_encoding).astype(float).fillna(0)
        
    
    logger.info("Removendo identificadores...")
    # Remover os identificadores que não serão utilizados como features
    X_train_reduced = X_train.drop(columns=['userId', 'pageId'], errors='ignore')
    X_test_reduced = X_test.drop(columns=['userId', 'pageId'], errors='ignore')
    
    trusted_data = {
        'X_train': X_train_reduced,
        'X_test': X_test_reduced,
        'y_train': y_train['TARGET'],
        'y_test': y_test['TARGET'],
        'encoder_mapping': encoder_mapping
    }
    
    return trusted_data


def load_train_data():
    """Carrega os dados de treino."""
    X_train = pd.read_parquet("data/processed_data/train/X_train.parquet")
    y_train = pd.read_parquet("data/processed_data/train/y_train.parquet")
    return X_train, y_train

# OPCIONAL: Para auxiliar no processo de seleção de features, eliminando features com alta correlação
def feature_selection(
    suggested_feats, 
    df_target, 
    target_col='TARGET', 
    method="correlation", 
    drop_cols=None, 
    threshold=0.9, 
    k_best=10):
    """
    Realiza a seleção de features no dataframe suggested_feats.

    Parâmetros:
        suggested_feats (DataFrame): DataFrame com as features finais.
        df_target (DataFrame): DataFrame contendo o target.
        target_col (str): String com a coluna alvo de engajamento (variável dependente).
        method (str): Método de seleção. Pode ser:
                      - "correlation": remove features altamente correlacionadas.
                      - "univariate": seleciona as k melhores features via teste univariado (ANOVA).
        drop_cols (list): Lista de colunas a serem descartadas antes da seleção (ex.: identificadores).
        threshold (float): Limite de correlação para remoção de features redundantes (método "correlation").
        k_best (int): Número de melhores features a serem selecionadas (método "univariate").

    Retorna:
        DataFrame: DataFrame com as features selecionadas e a coluna target_col.
    """
    # Cria uma cópia do dataframe para não afetar o original
    df = suggested_feats.merge(df_target, on=['userId','pageId'])

    # Remove as colunas que não são consideradas para o processo de seleção
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
        # Identifica features que possuem alta correlação com alguma outra feature
        drop_features = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        X_selected = X.drop(columns=drop_features)
        logger.info(f"Features removidas por alta correlação: {drop_features}")
        return pd.concat([X_selected, y], axis=1)
    
    elif method == "univariate":
        # Seleção univariada utilizando o teste F (ANOVA) para regressão/classificação
        selector = SelectKBest(score_func=f_classif, k=k_best)
        selector.fit(X, y)
        mask = selector.get_support()  # Máscara booleana das features selecionadas
        selected_features = X.columns[mask]
        logger.info(f"Features selecionadas pelo método univariado: {list(selected_features)}")
        X_selected = X[selected_features]
        return pd.concat([X_selected, y], axis=1)
    
    else:
        raise ValueError("Método de seleção desconhecido. Escolha 'correlation' ou 'univariate'.")