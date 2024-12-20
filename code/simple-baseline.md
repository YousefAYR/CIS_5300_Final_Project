# **Simple Baseline Model**

This script implements a baseline Logistic Regression model for sentiment classification on two datasets: FACTOID Reddit and Georgetown Labeled Tweets. It processes raw data, prepares train/dev/test splits, and evaluates model performance using accuracy, precision, recall, and F1-score.

---

## **Files**
- `raw_data/`: Contains the raw datasets to be processed.
- `data/`: Contains processed datasets split into `train.csv`, `dev.csv`, and `test.csv`.
- `evaluate.py`: Script for model evaluation metrics (accuracy, precision, recall, F1-score, and confusion matrix).

---

## **Usage Instructions**

### **1. Install Required Libraries**
Run the following command in your environment:
```bash
!pip install scikit-learn pandas numpy gdown
```

## 2. Download the Repository
Ensure the repository, including all raw datasets, is downloaded as a `.zip` file:

```bash
!unzip /path/to/CIS_5300_Final_Project-main.zip -d /content/
%cd /content/CIS_5300_Final_Project-main/
```

### **3. Run the Script**
Run the script to:

- Preprocess raw datasets (Georgetown, FACTOID, 2020 Tweets).
- Train a Logistic Regression model.
- Evaluate the model using the `evaluate.py` script.

## **Key Functions**

### **Preprocessing**
- **`determine_stance()`**: Converts dataset labels into numeric stances (`-1`, `0`, `1`).
- Splits data into `train`, `dev`, and `test`.

### **Training**
- **`train_baseline_model()`**: Trains a Logistic Regression model with TF-IDF features.

### **Evaluation**
- **`evaluate_model()`**: Computes accuracy, precision, recall, F1-score, and confusion matrix.
- **`display_evaluation_results()`**: Displays the evaluation metrics.


## **How to Use**

### **Preprocess the Data**
1. Download raw datasets via `gdown` and organize them into directories:
   - `raw_data/labeled_tweets_georgetown/`
   - `raw_data/factoid_reddit/`
   - `raw_data/2020_tweets/`

2. Run preprocessing for each dataset:
   ```python
   # Preprocess Georgetown Dataset
   Processed files saved in `data/labeled_tweets_georgetown/`.

   # Preprocess FACTOID Dataset
   Processed files saved in `data/factoid_reddit/`.

   # Preprocess 2020 Tweets Dataset
   Processed files saved in `data/2020_tweets/`.


## **Train the Baseline Model**
Run the `train_baseline_model` function for both datasets:
```python
factoid_model, factoid_vectorizer, factoid_y_test, factoid_y_test_pred = train_baseline_model(
    factoid_train, factoid_dev, factoid_test
)

georgetown_model, georgetown_vectorizer, georgetown_y_test, georgetown_y_test_pred = train_baseline_model(
    georgetown_train, georgetown_dev, georgetown_test
)
```

## **Evaluate the Model**

Use `evaluate.py` to evaluate and display results:

```python
# FACTOID Evaluation
factoid_labels = ['Negative', 'Neutral', 'Positive']
factoid_results = evaluate_model(factoid_y_test, factoid_y_test_pred, factoid_labels)
display_evaluation_results(factoid_results, factoid_labels)

# Georgetown Evaluation
georgetown_labels = ['Negative', 'Neutral', 'Positive']
georgetown_results = evaluate_model(georgetown_y_test, georgetown_y_test_pred, georgetown_labels)
display_evaluation_results(georgetown_results, georgetown_labels)
```


