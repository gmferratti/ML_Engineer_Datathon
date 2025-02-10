"""Utils Functions."""

import pandas as pd


def concatenate_csv_to_df(template_path: str, num_files: int) -> pd.DataFrame:
    """
    Concatena multiplos arquivos CSV.

    Args:
        template_path (str): Template do caminho dos arquivos CSV
            ex: "path/to/files/parte{}/file.csv".
        num_files (int): Número de arquivos a serem concatenados.

    Returns:
        pd.DataFrame: DataFrame concatenado com os dados de todos os arquivos.
    """
    concatenated_df = pd.DataFrame()

    for i in range(1, num_files + 1):
        file_path = template_path.format(i)
        df = pd.read_csv(file_path)
        concatenated_df = pd.concat([concatenated_df, df])

    return concatenated_df
