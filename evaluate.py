import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix

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

def display_evaluation_results(results, labels, output_file):
    """
    Display evaluation results in a readable format and write to a text file.

    Args:
    - results: Dictionary of evaluation metrics.
    - labels: List of class labels.
    - output_file: Name of the output text file.
    """
    with open(output_file, 'w') as f:
        # Write overall metrics
        f.write("Model Evaluation Metrics\n")
        f.write("-------------------------\n")
        f.write(f"Accuracy: {results['accuracy']:.4f}\n")
        f.write(f"Macro Precision: {results['macro_precision']:.4f}\n")
        f.write(f"Macro Recall: {results['macro_recall']:.4f}\n")
        f.write(f"Macro F1-Score: {results['macro_f1']:.4f}\n\n")

        # Write per-class metrics
        f.write("Per-Class Metrics:\n")
        for label, precision, recall, f1 in zip(labels, results['precision_per_class'], results['recall_per_class'], results['f1_per_class']):
            f.write(f"Class {label}: Precision={precision:.4f}, Recall={recall:.4f}, F1-Score={f1:.4f}\n")

        # Write confusion matrix
        f.write("\nConfusion Matrix:\n")
        cm = results['confusion_matrix']
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        f.write(cm_df.to_string())
        f.write("\n")
