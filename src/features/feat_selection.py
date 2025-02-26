import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif


def feature_selection(final_feats, target, method="correlation", drop_cols=None, threshold=0.9, k_best=10):
    """
    Realiza a seleção de features no dataframe final_feats.

    Parâmetros:
        final_feats (DataFrame): DataFrame com as features finais.
        target (str): Nome da coluna alvo (variável dependente).
        method (str): Método de seleção. Pode ser:
                      - "correlation": remove features altamente correlacionadas.
                      - "univariate": seleciona as k melhores features via teste univariado (ANOVA).
        drop_cols (list): Lista de colunas a serem descartadas antes da seleção (ex.: identificadores).
        threshold (float): Limite de correlação para remoção de features redundantes (método "correlation").
        k_best (int): Número de melhores features a serem selecionadas (método "univariate").

    Retorna:
        DataFrame: DataFrame com as features selecionadas e a coluna target.
    """
    # Cria uma cópia do dataframe para não afetar o original
    df = final_feats.copy()

    # Remove as colunas que não são consideradas para o processo de seleção
    if drop_cols is not None:
        df = df.drop(columns=drop_cols)
    
    # Separa as variáveis preditoras (X) e a variável alvo (y)
    X = df.drop(columns=[target])
    y = df[target]
    
    if method == "correlation":
        # Calcula a matriz de correlação entre as features
        corr_matrix = X.corr().abs()
        # Considera apenas a parte superior da matriz para evitar redundância
        upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        # Identifica features que possuem alta correlação com alguma outra feature
        drop_features = [col for col in upper_tri.columns if any(upper_tri[col] > threshold)]
        X_selected = X.drop(columns=drop_features)
        print(f"Features removidas por alta correlação: {drop_features}")
        return pd.concat([X_selected, y], axis=1)
    
    elif method == "univariate":
        # Seleção univariada utilizando o teste F (ANOVA) para regressão/classificação
        selector = SelectKBest(score_func=f_classif, k=k_best)
        selector.fit(X, y)
        mask = selector.get_support()  # Máscara booleana das features selecionadas
        selected_features = X.columns[mask]
        print(f"Features selecionadas pelo método univariado: {list(selected_features)}")
        X_selected = X[selected_features]
        return pd.concat([X_selected, y], axis=1)
    
    else:
        raise ValueError("Método de seleção desconhecido. Escolha 'correlation' ou 'univariate'.")
