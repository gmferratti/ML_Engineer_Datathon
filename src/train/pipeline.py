import os
import pandas as pd
import pickle
from utils import prepare_features, load_train_data
from config import logger
from recomendation_model.base_model import LightGBMRanker

def save_dataframe_as_parquet(df: pd.DataFrame, file_path: str) -> None:
    """
    Cria o diretório (caso não exista) e salva o DataFrame em formato Parquet.
    
    Args:
        df (pd.DataFrame): DataFrame a ser salvo.
        file_path (str): Caminho completo para salvar o arquivo Parquet.
    """
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_parquet(file_path)
    rel_path = os.path.relpath(file_path)
    logger.info("Arquivo salvo: %s", rel_path)


def train_model() -> None:
    """
    Pipeline de treinamento do modelo:
      1. Carrega o DataFrame final com features e target.
      2. Prepara os conjuntos de treino e teste e salva os dados em formato Parquet.
      3. Valida o carregamento dos dados.
      4. Treina o modelo LightGBMRanker e salva o modelo treinado em formato pickle.
      5. Valida o salvamento do modelo recarregando o arquivo pickle.
    """
    # 1. Carregar features finais com target
    final_feats_file = os.path.join("data", "features", "final_feats_with_target.parquet")
    logger.info("Carregando features finais com target de %s...", final_feats_file)
    final_feats = pd.read_parquet(final_feats_file)

    # 2. Preparar conjuntos de treino e teste
    logger.info("Preparando os conjuntos de treino e teste...")
    trusted_data = prepare_features(final_feats)

    # Criar diretório base para salvar os dados de treino
    train_base_path = os.path.join("data", "train")
    os.makedirs(train_base_path, exist_ok=True)

    # Salvar cada item de trusted_data em formato Parquet
    for key, data in trusted_data.items():
        # Converter para DataFrame, se necessário
        if key == "encoder_mapping" and isinstance(data, dict):
            data = pd.DataFrame(data)
        elif not isinstance(data, pd.DataFrame):
            try:
                data = pd.DataFrame(data)
            except Exception as error:
                logger.warning("Não foi possível converter '%s' para DataFrame: %s", key, error)
                continue

        file_name = f"{key}.parquet"
        file_path = os.path.join(train_base_path, file_name)
        logger.info("Salvando '%s'...", file_name)
        save_dataframe_as_parquet(data, file_path)

    logger.info("Pipeline de preparação de features concluído!")

    # 3. Validar o carregamento dos dados de treino
    logger.info("Validando o carregamento dos dados...")
    X_train_val, y_train_val = load_train_data()
    logger.info("Dados carregados: X_train shape: %s, y_train shape: %s", X_train_val.shape, y_train_val.shape)

    # 4. Carregar os dados para treino do modelo
    logger.info("Carregando dados para treino do modelo...")
    # Assume que os arquivos X_train, y_train e group_train foram salvos no passo anterior
    x_train_path = os.path.join(train_base_path, "X_train.parquet")
    y_train_path = os.path.join(train_base_path, "y_train.parquet")
    group_train_path = os.path.join(train_base_path, "group_train.parquet")
    
    X_train = pd.read_parquet(x_train_path)
    y_train = pd.read_parquet(y_train_path)
    group_train = pd.read_parquet(group_train_path)
    logger.info("Dados para treino carregados: X_train shape: %s, y_train shape: %s, group_train shape: %s",
                X_train.shape, y_train.shape, group_train.shape)

    # 5. Treinar o modelo LightGBMRanker
    logger.info("Treinando o modelo LightGBMRanker...")
    model = LightGBMRanker()
    model.train(
        X_train.values,
        y_train.values.ravel(),
        group_train["groupCount"].values
    )

    # Salvar o modelo treinado em um arquivo pickle na pasta data/train
    model_path = os.path.join(train_base_path, "lightgbm_ranker.pkl")
    with open(model_path, "wb") as f:
        pickle.dump(model, f)
    logger.info("Modelo treinado e salvo em '%s'", model_path)

    # 6. Validar o salvamento recarregando o modelo
    with open(model_path, "rb") as f:
        model_loaded = pickle.load(f)
    logger.info("Modelo carregado com sucesso!")

if __name__ == "__main__":
    train_model()
