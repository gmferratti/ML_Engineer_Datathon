import pandas as pd
from typing import List


def load_csv_files(file_paths: List[str]) -> pd.DataFrame:
    """
    Lê múltiplos arquivos CSV e combina em um único DataFrame.

    Args:
        file_paths (List[str]): Lista de caminhos dos arquivos CSV.

    Returns:
        pd.DataFrame: DataFrame combinado contendo os dados dos arquivos.
    """
    data_frames = [pd.read_csv(file) for file in file_paths]
    return pd.concat(data_frames, ignore_index=True)


def clean_train_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa e explode os dados de treino.

    Args:
        input_df (pd.DataFrame): DataFrame de treino.

    Returns:
        pd.DataFrame: DataFrame limpo e explodido.
    """
    df_treino = input_df.copy()

    df_treino = df_treino.drop(columns=["timestampHistory_new"])

    # Explodindo as colunas de historico para cada linha significar uma noticia
    USERS_COLS_TO_EXPLODE = [
        "history",
        "timestampHistory",
        "numberOfClicksHistory",
        "timeOnPageHistory",
        "scrollPercentageHistory",
        "pageVisitsCountHistory",
    ]

    for col in USERS_COLS_TO_EXPLODE:
        df_treino[col] = df_treino[col].str.split(", ")

    df_exploded = df_treino.explode([
        "history",
        "timestampHistory",
        "numberOfClicksHistory",
        "timeOnPageHistory",
        "scrollPercentageHistory",
        "pageVisitsCountHistory"
    ]).reset_index(drop=True)

    # Convertendo os tipos das colunas
    df_exploded["timestampHistory"] = pd.to_datetime(
        pd.to_numeric(
            df_exploded["timestampHistory"],
            errors="coerce"
        ),
        unit="ms"
    )
    df_exploded["numberOfClicksHistory"] = pd.to_numeric(
        df_exploded["numberOfClicksHistory"], errors="coerce")
    df_exploded["timeOnPageHistory"] = pd.to_numeric(
        df_exploded["timeOnPageHistory"], errors="coerce")
    df_exploded["scrollPercentageHistory"] = pd.to_numeric(
        df_exploded["scrollPercentageHistory"], errors="coerce")
    df_exploded["pageVisitsCountHistory"] = pd.to_numeric(
        df_exploded["pageVisitsCountHistory"], errors="coerce")

    df_exploded = df_exploded.rename(columns={
        "history": "historyId"
    })

    return df_exploded


def clean_items_data(input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Limpa os dados de itens e prepara para o merge com os dados de treino.

    Args:
        input_df (pd.DataFrame): DataFrame contendo os dados de itens.

    Returns:
        pd.DataFrame: DataFrame limpo e preparado.
    """
    df_itens = input_df.copy()

    # Garantir unicidade de 'page' e renomear para 'historyId'
    df_itens = df_itens.rename(columns={"page": "historyId"})

    # Selecionar apenas as colunas relevantes
    df_itens = df_itens[["historyId", "title", "body", "caption"]]

    return df_itens
