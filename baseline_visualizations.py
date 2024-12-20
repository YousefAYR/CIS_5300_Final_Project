
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Metrics from the evaluation results
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
factoid_values = [0.5783, 0.5418, 0.5783, 0.4859]
georgetown_values = [0.6788, 0.6945, 0.6788, 0.6509]

# Confusion matrices from the evaluation results
factoid_confusion_matrix = np.array([[15284, 838, 0], [5211, 1158, 0], [0, 0, 0]])
georgetown_confusion_matrix = np.array([[84, 4, 0], [25, 14, 0], [0, 0, 0]])

# Class labels
classes = ['Negative', 'Neutral', 'Positive']

# Visualization 1: Bar Chart for Overall Metrics Comparison
plt.figure(figsize=(10, 6))
x = np.arange(len(metrics))
width = 0.35

plt.bar(x - width / 2, factoid_values, width, label='FACTOID')
plt.bar(x + width / 2, georgetown_values, width, label='Georgetown')

plt.xticks(x, metrics)
plt.ylabel('Scores')
plt.title('Overall Metrics Comparison')
plt.legend()
plt.tight_layout()
plt.show()

# Visualization 2: Heatmaps for Confusion Matrices
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

sns.heatmap(factoid_confusion_matrix, annot=True, fmt="d", cmap="Blues", ax=axs[0], xticklabels=classes, yticklabels=classes)
axs[0].set_title('FACTOID Confusion Matrix')
axs[0].set_xlabel('Predicted')
axs[0].set_ylabel('Actual')

sns.heatmap(georgetown_confusion_matrix, annot=True, fmt="d", cmap="Greens", ax=axs[1], xticklabels=classes, yticklabels=classes)
axs[1].set_title('Georgetown Confusion Matrix')
axs[1].set_xlabel('Predicted')
axs[1].set_ylabel('Actual')

plt.tight_layout()
plt.show()

# Visualization 3: Class-wise Metrics for Each Dataset
factoid_class_metrics = {
    'Negative': [0.59, 0.94, 0.72],
    'Neutral': [0.49, 0.13, 0.21],
    'Positive': [0.45, 0.04, 0.07],
}

georgetown_class_metrics = {
    'Negative': [0.78, 0.41, 0.54],
    'Neutral': [0.66, 0.91, 0.77],
    'Positive': [0.70, 0.36, 0.47],
}

metrics_per_class = ['Precision', 'Recall', 'F1-Score']
x = np.arange(len(metrics_per_class))

fig, axs = plt.subplots(1, 2, figsize=(14, 6), sharey=True)
for i, (dataset, metrics_data) in enumerate([("FACTOID", factoid_class_metrics), ("Georgetown", georgetown_class_metrics)]):
    for j, (cls, values) in enumerate(metrics_data.items()):
        axs[i].bar(x + j * 0.25, values, width=0.25, label=f'{cls}')
    axs[i].set_title(f'{dataset} Class-wise Metrics')
    axs[i].set_xticks(x + 0.25)
    axs[i].set_xticklabels(metrics_per_class)
    axs[i].set_ylabel('Scores')
    axs[i].legend(title='Class')

plt.tight_layout()
plt.show()
