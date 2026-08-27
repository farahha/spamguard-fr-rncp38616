import joblib
import hashlib

from spamguard.config import PROJECT_ROOT
from spamguard.data import deduplicate_french_text, load_frozen_split, load_processed_data
from spamguard.ml import build_ml_pipelines


def test_historical_split_is_unchanged_and_has_no_text_leakage():
    data = deduplicate_french_text(load_processed_data())
    split_path = PROJECT_ROOT / "data" / "processed" / "ml_split_ids.csv"
    digest = hashlib.sha256(split_path.read_bytes()).hexdigest()
    assert digest == "6d1557cb761f1b271f362767ece08dedc55b2f56d28f496bded8dad4b0c97f87"
    train, test = load_frozen_split(data)
    assert len(train) == 4107
    assert len(test) == 1027
    assert set(train["text_fr"]).isdisjoint(test["text_fr"])


def test_pipelines_are_importable():
    assert set(build_ml_pipelines()) == {"logistic_regression", "linear_svm", "random_forest"}


def test_saved_pipeline_loads_and_predicts_valid_classes():
    model = joblib.load(PROJECT_ROOT / "models" / "ml_spam_classifier.joblib")
    predictions = model.predict(["Votre colis est arrivé.", "Gagnez 1000 euros maintenant !"])
    assert len(predictions) == 2
    assert set(predictions).issubset({"ham", "spam"})
