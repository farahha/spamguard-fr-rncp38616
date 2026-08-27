import sys
from pathlib import Path

import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from spamguard.inference import load_ml_model, predict_ml  # noqa: E402


@st.cache_resource
def get_model():
    return load_ml_model()


st.set_page_config(page_title="SpamGuard-FR — Bloc 3", page_icon="🛡️")
st.title("SpamGuard-FR")
st.subheader("Démonstration Machine Learning — Linear SVM")
message = st.text_area("SMS à analyser", height=160)
if st.button("Analyser le message", type="primary"):
    try:
        result = predict_ml(message, get_model())
    except ValueError as error:
        st.warning(str(error))
    else:
        if result["label"] == "spam":
            st.error("SPAM — message suspect")
        else:
            st.success("HAM — message légitime")
        st.caption(
            f"Score de décision SVM : {result['decision_score']:.3f}. "
            "Distance à la frontière, et non probabilité."
        )
st.caption("Corpus français traduit automatiquement ; résultat indicatif.")
