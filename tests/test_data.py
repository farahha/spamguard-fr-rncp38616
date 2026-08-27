from spamguard.data import PROCESSED_COLUMNS, load_processed_data


def test_dataset_is_loaded():
    assert not load_processed_data().empty


def test_processed_columns_are_exact():
    assert load_processed_data().columns.tolist() == PROCESSED_COLUMNS


def test_labels_are_expected():
    assert set(load_processed_data()["label"].unique()) == {"ham", "spam"}


def test_french_text_is_not_empty():
    text = load_processed_data()["text_fr"]
    assert not text.isna().any()
    assert text.astype(str).str.strip().ne("").all()
