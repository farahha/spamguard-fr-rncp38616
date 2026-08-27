import re

import pandas as pd


URL_PATTERN = re.compile(r"(?:https?://|www\.|\b[\w.-]+\.(?:com|fr|net|org)\b)", re.I)


def uppercase_ratio(text: str) -> float:
    letters = [char for char in text if char.isalpha()]
    return sum(char.isupper() for char in letters) / len(letters) if letters else 0.0


def add_exploratory_features(data: pd.DataFrame) -> pd.DataFrame:
    """Ajoute des variables uniquement destinées à l'EDA."""
    enriched = data.copy()
    text = enriched["text_fr"].fillna("").astype(str)
    enriched["n_chars"] = text.str.len()
    enriched["n_words"] = text.str.split().str.len()
    enriched["has_url"] = text.str.contains(URL_PATTERN, regex=True)
    enriched["has_number"] = text.str.contains(r"\d", regex=True)
    enriched["n_special_chars"] = text.map(
        lambda value: sum(not char.isalnum() and not char.isspace() for char in value)
    )
    enriched["uppercase_ratio"] = text.map(uppercase_ratio)
    return enriched
