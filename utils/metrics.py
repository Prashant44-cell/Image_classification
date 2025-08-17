# deep_image_analyzer/utils/metrics.py

# Import scikit-learn metrics
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def classification_metrics(y_true, y_pred):
    """
    Compute overall accuracy, precision, recall, and F1-score.
    """
    acc = accuracy_score(y_true, y_pred)   # Weighted accuracy
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="weighted"
    )
    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1
    }
