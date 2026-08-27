"""Inférence du modèle Machine Learning validé du Bloc 3."""

from pathlib import Path
from typing import Any

import joblib
import numpy as np

from spamguard.config import PROJECT_ROOT


ML_MODEL_PATH = PROJECT_ROOT / "models" / "ml_spam_classifier.joblib"


def validate_message(text: str) -> str:
    if not isinstance(text, str) or not text.strip():
        raise ValueError("Saisissez un SMS avant de lancer l'analyse.")
    return text.strip()


def load_ml_model(path: Path | str = ML_MODEL_PATH) -> Any:
    model_path = Path(path)
    if not model_path.is_file():
        raise FileNotFoundError(f"Modèle ML introuvable : {model_path}")
    model = joblib.load(model_path)
    if not hasattr(model, "predict") or not hasattr(model, "decision_function"):
        raise TypeError("Le pipeline Linear SVM attendu n'est pas disponible.")
    return model


def predict_ml(text: str, model: Any | None = None) -> dict:
    message = validate_message(text)
    active_model = model if model is not None else load_ml_model()
    label = str(active_model.predict([message])[0])
    if label not in {"ham", "spam"}:
        raise ValueError(f"Classe inattendue : {label}")
    score = float(np.asarray(active_model.decision_function([message])).reshape(-1)[0])
    return {"label": label, "decision_score": score}
