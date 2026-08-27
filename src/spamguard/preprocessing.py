import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Réduit les suites d'espaces, sans servir encore au dataset d'audit."""
    return re.sub(r"\s+", " ", text).strip()


def lowercase(text: str) -> str:
    """Passe un texte en minuscules (fonction préparatoire, non appliquée)."""
    return text.lower()


def strip_accents(text: str) -> str:
    """Retire les accents (fonction expérimentale, non appliquée)."""
    normalized = unicodedata.normalize("NFD", text)
    return "".join(char for char in normalized if unicodedata.category(char) != "Mn")
