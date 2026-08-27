from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from spamguard.config import PROJECT_ROOT, load_config


PROCESSED_COLUMNS = ["id", "label", "text_fr", "text_en"]


def load_raw_data(path: Path | None = None) -> pd.DataFrame:
    """Charge le CSV Hugging Face sans transformer son contenu."""
    config = load_config()
    raw_path = path or PROJECT_ROOT / config["paths"]["raw_data"]
    return pd.read_csv(raw_path)


def build_french_dataset(raw: pd.DataFrame) -> pd.DataFrame:
    """Sélectionne et renomme seulement les colonnes utiles.

    L'identifiant dépend de la position stable de la ligne dans le fichier source.
    Le texte et les labels ne sont ni nettoyés ni corrigés.
    """
    required = ["labels", "text_fr", "text"]
    missing = sorted(set(required) - set(raw.columns))
    if missing:
        raise ValueError(f"Colonnes source manquantes : {missing}")
    processed = raw.loc[:, required].rename(
        columns={"labels": "label", "text": "text_en"}
    )
    processed.insert(0, "id", [f"sgfr-{i:06d}" for i in range(len(processed))])
    return processed.loc[:, PROCESSED_COLUMNS]


def save_processed_data(data: pd.DataFrame, path: Path | None = None) -> Path:
    config = load_config()
    output_path = path or PROJECT_ROOT / config["paths"]["processed_data"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def prepare_dataset() -> Path:
    return save_processed_data(build_french_dataset(load_raw_data()))


def load_processed_data(path: Path | None = None) -> pd.DataFrame:
    config = load_config()
    processed_path = path or PROJECT_ROOT / config["paths"]["processed_data"]
    return pd.read_csv(processed_path)


def deduplicate_french_text(data: pd.DataFrame) -> pd.DataFrame:
    """Conserve la première occurrence de chaque texte français strictement identique."""
    conflicts = data.groupby("text_fr", dropna=False)["label"].nunique()
    if (conflicts > 1).any():
        raise ValueError("Des textes français identiques ont des labels contradictoires.")
    return data.drop_duplicates(subset=["text_fr"], keep="first").reset_index(drop=True)


def save_deduplicated_data(data: pd.DataFrame, path: Path | None = None) -> Path:
    config = load_config()
    output_path = path or PROJECT_ROOT / config["paths"]["deduplicated_data"]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    data.to_csv(output_path, index=False)
    return output_path


def create_stratified_split(
    data: pd.DataFrame,
    *,
    test_size: float = 0.20,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Crée l'unique split stratifié utilisé par tous les modèles classiques."""
    train, test = train_test_split(
        data,
        test_size=test_size,
        random_state=random_state,
        stratify=data["label"],
    )
    return train.reset_index(drop=True), test.reset_index(drop=True)


def save_split_ids(train: pd.DataFrame, test: pd.DataFrame, path: Path | None = None) -> Path:
    config = load_config()
    output_path = path or PROJECT_ROOT / config["paths"]["split_ids"]
    split_ids = pd.concat(
        [
            train[["id"]].assign(split="train"),
            test[["id"]].assign(split="test"),
        ],
        ignore_index=True,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    split_ids.to_csv(output_path, index=False)
    return output_path


def load_frozen_split(
    data: pd.DataFrame,
    path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Recharge le split ML existant sans le recalculer ni le modifier."""
    config = load_config()
    split_path = path or PROJECT_ROOT / config["paths"]["split_ids"]
    split_ids = pd.read_csv(split_path)
    if split_ids.columns.tolist() != ["id", "split"]:
        raise ValueError("Le fichier de split doit contenir exactement les colonnes id et split.")
    if split_ids["id"].duplicated().any():
        raise ValueError("Le fichier de split contient des identifiants dupliqués.")
    if set(split_ids["split"]) != {"train", "test"}:
        raise ValueError("Le fichier de split contient une partition inconnue.")
    if set(split_ids["id"]) != set(data["id"]):
        raise ValueError("Les identifiants du split ne correspondent pas au dataset dédupliqué.")
    assigned = split_ids.merge(data, on="id", how="left", validate="one_to_one", sort=False)
    train = assigned.loc[assigned["split"] == "train", data.columns].reset_index(drop=True)
    test = assigned.loc[assigned["split"] == "test", data.columns].reset_index(drop=True)
    return train, test


if __name__ == "__main__":
    print(prepare_dataset())
