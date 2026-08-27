import pytest

from spamguard.inference import load_ml_model, predict_ml, validate_message


@pytest.fixture(scope="module")
def model():
    return load_ml_model()


def test_ml_model_loads_and_predicts(model):
    result = predict_ml("Rendez-vous demain à 10 heures.", model)
    assert result["label"] in {"ham", "spam"}
    assert isinstance(result["decision_score"], float)


@pytest.mark.parametrize(
    "message",
    ["Élodie arrive à 18 h 😊", "https://exemple.fr 0601020304", "ok", "long " * 500],
)
def test_robust_inputs(message, model):
    assert predict_ml(message, model)["label"] in {"ham", "spam"}


@pytest.mark.parametrize("message", ["", "   ", "\n\t"])
def test_empty_message_is_rejected(message, model):
    with pytest.raises(ValueError, match="Saisissez un SMS"):
        validate_message(message)
    with pytest.raises(ValueError, match="Saisissez un SMS"):
        predict_ml(message, model)
