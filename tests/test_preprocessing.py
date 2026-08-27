from spamguard.preprocessing import lowercase, normalize_whitespace, strip_accents


def test_normalize_whitespace_preserves_punctuation_and_accents():
    assert normalize_whitespace("  Déjà\n  vu !  ") == "Déjà vu !"


def test_lowercase():
    assert lowercase("SMS Gratuit") == "sms gratuit"


def test_strip_accents_is_explicit():
    assert strip_accents("été") == "ete"
