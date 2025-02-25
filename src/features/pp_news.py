"""Preprocessing for news features."""

import re

import pandas as pd
import re
import nltk
import unicodedata

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from utils import concatenate_csv_to_df

from constants import (
    COLS_TO_CLEAN,
    COLS_TO_DROP)

from feat_settings import (
    SAMPLE_RATE,
    NEWS_TEMP_PATH,
    NEWS_N_CSV_FILES
)

# Downloading extra dependencies
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
print("Downloaded NLTK dependencies.")

def preprocess_news() -> pd.DataFrame:
    """
    Realiza o pré-processamento dos dados de notícias:
    - Concatena CSVs.
    - Extrai informações da URL (localidade, tema da notícia).
    - Limpa colunas de texto.
    - Remove colunas desnecessárias.
    """
    # Concatena CSVs
    df_news = concatenate_csv_to_df(NEWS_TEMP_PATH, NEWS_N_CSV_FILES)
    
    # Faz o sampling dos dados
    df_news = df_news.sample(frac=SAMPLE_RATE, random_state=42)

    # Renomeia coluna de chave primária
    df_news = df_news.rename(columns={"page": "pageId"})

    # Converte colunas de data de publicação e modificação
    for col in ["issued", "modified"]:
        df_news[col] = pd.to_datetime(df_news[col])
        df_news[f"{col}Date"] = df_news[col].dt.date
        df_news[f"{col}Time"] = df_news[col].dt.time

    # Extrai informações do miolo da URL
    df_news["urlExtracted"] = df_news["url"].apply(_extract_url_midsection)

    # Extrai localidade da URL
    df_news['local'] = df_news['urlExtracted'].apply(_extract_location)
    df_news['localState'] = df_news['local'].str.split('/').str[0]
    df_news['localRegion'] = df_news['local'].str.split('/').str[1]
    
    # Extrai tema da notícia da URL
    df_news['theme'] = df_news['urlExtracted'].apply(_extract_theme)
    df_news['themeMain'] = df_news['theme'].str.split('/').str[0]
    df_news['themeSub'] = df_news['theme'].str.split('/').str[1]
    
    # Limpa colunas de texto
    # for col in COLS_TO_CLEAN:
    #     df_news[f"{col}Cleaned"] = df_news[col].apply(_preprocess_text)
    
    # Obs. Não estamos usando as colunas de texto para nada, somente URL
    
    # Remove colunas desnecessárias
    df_news = df_news.drop(columns=COLS_TO_DROP)

    return df_news


def _extract_url_midsection(url):
    """Extrai o miolo relevante da URL."""
    regex = r"(?<=g1\.globo\.com\/)(.*?)(?=\/noticia)"
    match = re.search(regex, url)
    return match.group() if match else None


def _extract_location(url_part):
    """Extrai a localidade a partir do miolo da URL."""
    if not url_part:
        return None
    regex = re.compile(r"^[a-z]{2}/[a-z-]+")
    match = regex.match(url_part)
    return match.group() if match else None


def _extract_theme(url_part):
    """Extrai o tema da notícia a partir do miolo da URL."""
    location = _extract_location(url_part)
    if pd.notna(url_part):
        if location:
            theme = url_part.replace(location, "").lstrip("/")
            return theme if theme else None
        else:
            return url_part
    return None


def _preprocess_text(text):
    """Padroniza e limpa o texto de notícias."""
    if not isinstance(text, str):
        return ""
    
    # Remover acentos
    text = unicodedata.normalize('NFKD', text).encode('ascii', 'ignore').decode('utf-8')
    
    # Remover caracteres especiais e números
    text = re.sub(r'\W+', ' ', text)
    text = re.sub(r'\d+', '', text)
    
    # Converter para minúsculas
    text = text.lower()
    
    # Tokenização
    words = text.split()
    
    # Remover stopwords das tokens
    stop_words = set(stopwords.words('portuguese'))
    words = [word for word in words if word not in stop_words]
    
    # Lemmatização
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words]
    
    return ' '.join(words) # junta novamente as palavras