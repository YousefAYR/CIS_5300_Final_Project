import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model(y_true, y_pred, labels=None):
    """
    Evaluate model performance.
    
    Args:
    - y_true: List or numpy array of true labels.
    - y_pred: List or numpy array of predicted labels.
    - labels: List of label names.
    
    Returns:
    - A dictionary containing evaluation metrics.
    """
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    
    # Precision, Recall, F1-Score
    precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)
    recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)
    
    # Classification Report
    report = classification_report(y_true, y_pred, target_names=labels, zero_division=0)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=range(len(labels)))
    
    return {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'classification_report': report,
        'confusion_matrix': cm
    }

def display_evaluation_results(results, labels):
    """
    Display evaluation results in a readable format.
    
    Args:
    - results: Dictionary of evaluation metrics.
    - labels: List of label names.
    """
    print("Model Evaluation Metrics")
    print("-------------------------")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-Score: {results['f1_score']:.4f}\n")
    
    print("Classification Report")
    print(results['classification_report'])
    
    print("Confusion Matrix")
    cm = results['confusion_matrix']
    df_cm = pd.DataFrame(cm, index=labels, columns=labels)
    print(df_cm)

# Example usage
if __name__ == "__main__":
    # Example: Replace with  true and predicted labels
    y_true = [0, 1, 0, 1, 2, 2, 1, 0]
    y_pred = [0, 1, 0, 0, 2, 2, 1, 1]
    labels = ['Negative', 'Neutral', 'Positive']
    
    results = evaluate_model(y_true, y_pred, labels)
    display_evaluation_results(results, labels)
