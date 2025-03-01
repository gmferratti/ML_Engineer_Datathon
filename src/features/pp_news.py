import re
import unicodedata
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from .utils import concatenate_csv_files
from .constants import NEWS_COLS_TO_DROP
from src.config import logger, NEWS_DIRECTORY


def preprocess_news(selected_pageIds: pd.Series) -> pd.DataFrame:
    """
    Pré-processa dados de notícias.

    Args:
        selected_pageIds (pd.Series): Lista de pageId.

    Returns:
        pd.DataFrame: Notícias pré-processadas.
    """
    _download_resource("stopwords", ["corpora/stopwords"])
    _download_resource("wordnet", ["corpora/wordnet", "corpora/wordnet.zip"])
    _download_resource("omw-1.4", ["corpora/omw-1.4", "corpora/omw-1.4.zip"])
    df = concatenate_csv_files(NEWS_DIRECTORY)
    df = df.rename(columns={"page": "pageId"})
    df = df[df["pageId"].isin(selected_pageIds)]
    for col in ["issued", "modified"]:
        df[col] = pd.to_datetime(df[col])
        df[f"{col}Date"] = df[col].dt.date
        df[f"{col}Time"] = df[col].dt.time
    df["urlExtracted"] = df["url"].apply(_extract_url_mid_section)
    df["local"] = df["urlExtracted"].apply(_extract_location)
    df["localState"] = df["local"].str.split("/").str[0]
    df["localRegion"] = df["local"].str.split("/").str[1]
    df["theme"] = df["urlExtracted"].apply(_extract_theme)
    df["themeMain"] = df["theme"].str.split("/").str[0]
    df["themeSub"] = df["theme"].str.split("/").str[1]
    return df.drop(columns=NEWS_COLS_TO_DROP)


def _download_resource(resource_name: str, resource_paths: list) -> None:
    """
    Verifica e baixa um recurso NLTK se necessário.

    Args:
        resource_name (str): Nome do recurso.
        resource_paths (list): Caminhos para verificação.
    """
    found = False
    for path in resource_paths:
        try:
            nltk.data.find(path)
            found = True
            logger.info("Recurso '%s' já baixado.", resource_name)
            break
        except LookupError:
            continue
    if not found:
        nltk.download(resource_name)
        logger.info("Recurso '%s' baixado.", resource_name)


def _extract_url_mid_section(url: str) -> str:
    """
    Extrai a parte da URL entre 'g1.globo.com/' e '/noticia'.

    Args:
        url (str): URL completa.

    Returns:
        str: Miolo da URL ou None.
    """
    regex = r"(?<=g1\.globo\.com\/)(.*?)(?=\/noticia)"
    match = re.search(regex, url)
    return match.group() if match else None


def _extract_location(url_part: str) -> str:
    """
    Extrai a localidade do miolo da URL.

    Args:
        url_part (str): Miolo da URL.

    Returns:
        str: Localidade ou None.
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
        url_part (str): Miolo da URL.

    Returns:
        str: Tema ou None.
    """
    if not url_part:
        return None
    loc = _extract_location(url_part)
    if loc:
        theme = url_part.replace(loc, "").lstrip("/")
        return theme if theme else None
    return url_part


def _preprocess_text(text: str) -> str:
    """
    Limpa e padroniza o texto removendo acentos, caracteres especiais,
    números, convertendo para minúsculas e removendo stopwords.

    Args:
        text (str): Texto original.

    Returns:
        str: Texto processado.
    """
    if not isinstance(text, str):
        return ""
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("utf-8")
    text = re.sub(r"\W+", " ", text)
    text = re.sub(r"\d+", "", text)
    text = text.lower()
    words = text.split()
    stop_words = set(stopwords.words("portuguese"))
    words = [w for w in words if w not in stop_words]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(w) for w in words]
    return " ".join(words)
