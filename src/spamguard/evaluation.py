PLANNED_METRICS = (
    "precision_spam",
    "recall_spam",
    "f1_spam",
    "f1_macro",
    "confusion_matrix",
)


def business_confusion_counts(y_true, y_pred) -> dict[str, int]:
    """Retourne TN, FP, FN et TP avec spam comme classe positive."""
    from sklearn.metrics import confusion_matrix

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=["ham", "spam"]).ravel()
    return {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
    }
