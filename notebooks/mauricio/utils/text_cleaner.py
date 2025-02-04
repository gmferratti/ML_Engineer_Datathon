import re
from nltk.corpus import stopwords
from unidecode import unidecode

stopwords_pt = set(stopwords.words('portuguese'))


def clean_pt_text(text: str) -> str:
    """
    Limpa texto em portugues:
    - Converte para minusculas
    - Remove acentuacao
    - Remove pontuacao
    - Remove digitos
    - Remove stopwords
    """
    if not isinstance(text, str):
        return ""

    text = text.lower()

    # Remove acentuacao
    text = unidecode(text)

    # Remove pontuacao e caracteres especiais (mantendo espacos e letras)
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)

    # Remove digitos
    text = re.sub(r'\d+', '', text)

    # Remove stopwords
    tokens = text.split()
    tokens = [t for t in tokens if t not in stopwords_pt]

    return " ".join(tokens)
