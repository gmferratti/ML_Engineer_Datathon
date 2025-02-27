"""Preprocessing for news features."""

import re
import unicodedata

import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

from utils import concatenate_csv_to_df
from constants import (
    NEWS_COLS_TO_CLEAN,
    NEWS_COLS_TO_DROP
)
from feat_settings import (
    SAMPLE_RATE,
    NEWS_TEMP_PATH,
    NEWS_N_CSV_FILES
)

# Downloading NLTK dependencies
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("Downloaded NLTK dependencies.")


def preprocess_news() -> pd.DataFrame:
    """
    Realiza o pré-processamento dos dados de notícias:
      1. Concatena múltiplos CSVs.
      2. Realiza amostragem (sampling).
      3. Renomeia a coluna de chave primária.
      4. Converte colunas de data e hora em tipos datetime.
      5. Extrai informações do 'miolo' da URL (localidade, tema da notícia).
      6. Remove colunas desnecessárias.

    Returns:
        pd.DataFrame: DataFrame resultante do pré-processamento.
    """
    # 1. Concatena CSVs
    df_news = concatenate_csv_to_df(NEWS_TEMP_PATH, NEWS_N_CSV_FILES)

    # 2. Realiza sampling dos dados
    df_news = df_news.sample(frac=SAMPLE_RATE, random_state=42)

    # 3. Renomeia a coluna de chave primária
    df_news = df_news.rename(columns={"page": "pageId"})

    # 4. Converte colunas de data/hora em tipos datetime
    for col in ["issued", "modified"]:
        df_news[col] = pd.to_datetime(df_news[col])
        df_news[f"{col}Date"] = df_news[col].dt.date
        df_news[f"{col}Time"] = df_news[col].dt.time

    # 5. Extrai informações do miolo da URL
    df_news["urlExtracted"] = df_news["url"].apply(_extract_url_mid_section)

    # 5.1 Extrai localidade a partir da URL
    df_news["local"] = df_news["urlExtracted"].apply(_extract_location)
    df_news["localState"] = df_news["local"].str.split("/").str[0]
    df_news["localRegion"] = df_news["local"].str.split("/").str[1]

    # 5.2 Extrai o tema da notícia
    df_news["theme"] = df_news["urlExtracted"].apply(_extract_theme)
    df_news["themeMain"] = df_news["theme"].str.split("/").str[0]
    df_news["themeSub"] = df_news["theme"].str.split("/").str[1]

    # (Opcional) Limpeza de colunas de texto.
    # Descomentar, caso for utilizar os textos no futuro.
    # for col in NEWS_COLS_TO_CLEAN:
    #     df_news[f"{col}Cleaned"] = df_news[col].apply(_preprocess_text)

    # 6. Remove colunas desnecessárias
    df_news = df_news.drop(columns=NEWS_COLS_TO_DROP)

    return df_news


def _extract_url_mid_section(url: str) -> str:
    """
    Extrai o miolo relevante da URL, entre 'g1.globo.com/' e '/noticia'.

    Args:
        url (str): URL completa da notícia.

    Returns:
        str: Parte da URL correspondente ao domínio de localidade/tema.
    """
    regex = r"(?<=g1\.globo\.com\/)(.*?)(?=\/noticia)"
    match = re.search(regex, url)
    return match.group() if match else None


def _extract_location(url_part: str) -> str:
    """
    Extrai a localidade a partir do miolo da URL.

    Args:
        url_part (str): Parte da URL extraída em `_extract_url_mid_section`.

    Returns:
        str: Cadeia de texto representando a localidade
             (ex.: 'sp/sao-paulo').
    """
    if not url_part:
        return None
    regex = re.compile(r"^[a-z]{2}/[a-z-]+")
    match = regex.match(url_part)
    return match.group() if match else None


def _extract_theme(url_part: str) -> str:
    """
    Extrai o tema da notícia a partir do miolo da URL.

    Args:
        url_part (str): Parte da URL extraída em `_extract_url_mid_section`.

    Returns:
        str: Tema encontrado (ex.: 'economia/mercados').
    """
    if not url_part:
        return None

    location = _extract_location(url_part)
    if location:
        theme = url_part.replace(location, "").lstrip("/")
        return theme if theme else None
    return url_part


def _preprocess_text(text: str) -> str:
    """
    Padroniza e limpa o texto de notícias:
      - Remove acentos.
      - Remove caracteres especiais e dígitos.
      - Converte para minúsculas.
      - Remove stopwords.
      - Aplica lematização.

    Args:
        text (str): Texto a ser limpo.

    Returns:
        str: Texto processado.
    """
    if not isinstance(text, str):
        return ""

    # Remove acentos
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')

    # Remove caracteres especiais e números
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)

    # Converte para minúsculas
    text = text.lower()

    # Tokenização
    words = text.split()

    # Remove stopwords
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in words if word not in stop_words]

    # Lematização
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]

    # Retorna o texto limpo
    return ' '.join(words)
