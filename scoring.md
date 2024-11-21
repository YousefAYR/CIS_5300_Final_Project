# **Evaluation Metrics for Stance Detection**

## **Overview**
The evaluation of our stance detection model is based on a combination of metrics designed to capture the overall performance and address class imbalances in the dataset. These metrics include **Accuracy**, **Precision**, **Recall**, and **F1-Score**, with a focus on the **Macro-Averaged F1-Score** to evaluate performance across all classes equitably.

## **Metrics Definitions**

### **1. Accuracy**
Accuracy measures the fraction of correctly classified instances:
\[
\text{Accuracy} = \frac{\text{Number of Correct Predictions}}{\text{Total Number of Predictions}}
\]
This metric provides an overall view of model performance but can be misleading in imbalanced datasets, which is why additional metrics are included.

---

### **2. Precision**
Precision measures the proportion of correctly predicted instances for each class relative to all predictions for that class:

$$
\text{Precision} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Positives}}
$$

---

### **3. Recall**
Recall (also known as Sensitivity) measures the proportion of actual instances of a class that were correctly predicted:
\[
\text{Recall} = \frac{\text{True Positives}}{\text{True Positives} + \text{False Negatives}}
\]

---

### **4. F1-Score**
F1-Score is the harmonic mean of Precision and Recall, balancing their trade-off:
\[
\text{F1-Score} = 2 \times \frac{\text{Precision} \times \text{Recall}}{\text{Precision} + \text{Recall}}
\]

---

### **5. Macro-Averaged Metrics**
Macro-averaging ensures that all classes are treated equally, regardless of their frequency in the dataset. For Precision, Recall, and F1-Score, the macro-average is calculated as:
\[
\text{Macro Metric} = \frac{1}{N} \sum_{i=1}^N \text{Metric for Class } i
\]
Where \(N\) is the number of classes.

This averaging method is particularly useful in imbalanced datasets, as it prevents the performance on majority classes from overshadowing the minority classes.

---

### **6. Confusion Matrix**
The confusion matrix provides a detailed breakdown of the model's predictions compared to actual values. It is a square matrix where:
- Rows represent the true class labels.
- Columns represent the predicted class labels.

Each cell \((i, j)\) indicates the number of instances with true label \(i\) predicted as \(j\).

---

## **Justification**
- The **Macro-Averaged F1-Score** is chosen as the primary metric for this task, as it is the standard metric used in shared tasks such as SemEval-2016 Task 6: *Detecting Stance in Tweets*.
- **Accuracy** complements F1-Score by providing a general overview of performance.
- The inclusion of **Precision** and **Recall** ensures a deeper understanding of the trade-offs between false positives and false negatives.

---

## **References**
1. SemEval-2016 Task 6: Detecting Stance in Tweets  
   Link: [SemEval 2016 Overview](https://www.aclweb.org/anthology/S16-1001.pdf)

2. Wikipedia article on F1-score  
   Link: [F1-Score - Wikipedia](https://en.wikipedia.org/wiki/F1_score)

3. Overview of Macro-Averaging in Text Classification  
   Link: [Text Classification Metrics - Macro vs Micro](https://medium.com/@ehudkr/a-visual-way-to-think-on-macro-and-micro-averages-in-classification-metrics-190285dc927f)
