import numpy as np
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, labels=None):
    """
    Evaluate model performance based on gold standard labels and predictions.

    Args:
    - y_true: List or numpy array of true labels.
    - y_pred: List or numpy array of predicted labels.
    - labels: List of class labels (e.g., [-1, 0, 1] for stance detection).

    Returns:
    - A dictionary containing evaluation metrics.
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)

    # Precision, Recall, F1-score
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average=None, zero_division=0
    )

    # Macro-average scores
    macro_precision, macro_recall, macro_f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=labels, average='macro', zero_division=0
    )

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=labels)

    return {
        'accuracy': accuracy,
        'macro_precision': macro_precision,
        'macro_recall': macro_recall,
        'macro_f1': macro_f1,
        'precision_per_class': precision,
        'recall_per_class': recall,
        'f1_per_class': f1,
        'confusion_matrix': cm
    }

def display_evaluation_results(results, labels):
    """
    Display evaluation results in a readable format.

    Args:
    - results: Dictionary of evaluation metrics.
    - labels: List of class labels.
    """
    print("Model Evaluation Metrics")
    print("-------------------------")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Macro Precision: {results['macro_precision']:.4f}")
    print(f"Macro Recall: {results['macro_recall']:.4f}")
    print(f"Macro F1-Score: {results['macro_f1']:.4f}\n")

    print("Per-Class Metrics:")
    for label, precision, recall, f1 in zip(labels, results['precision_per_class'], results['recall_per_class'], results['f1_per_class']):
        print(f"Class {label}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}")

    print("\nConfusion Matrix:")
    cm = results['confusion_matrix']
    cm_df = pd.DataFrame(cm, index=labels, columns=labels)
    print(cm_df)
