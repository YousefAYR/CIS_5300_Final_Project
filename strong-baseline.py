from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np
import pandas as pd
from evaluate import evaluate_model, display_evaluation_results

def to_ascii(text):
    text = str(text)
    return ''.join(i for i in text if ord(i) < 128)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
pretrained_LM_path = "kornosk/bert-election2020-twitter-stance-trump-KE-MLM"

tokenizer = AutoTokenizer.from_pretrained(pretrained_LM_path)
model = AutoModelForSequenceClassification.from_pretrained(pretrained_LM_path)

tweet_test_file = 'data/labeled_tweets_georgetown/test.csv'
reddit_test_file = 'data/factoid_reddit/test.csv'

df_t_test = pd.read_csv(tweet_test_file)
df_r_test = pd.read_csv(reddit_test_file)

df_t_test['text'] = df_t_test['text'].apply(to_ascii)
df_r_test['text'] = df_r_test['text'].apply(to_ascii)


id2label = {
    0: "AGAINST",
    1: "FAVOR",
    2: "NONE"
}
label2value = {
    "AGAINST": -1,
    "FAVOR": 1,
    "NONE": 0
}

def predict_label(text):
    inputs = tokenizer(text.lower(), return_tensors="pt")
    outputs = model(**inputs)
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
    predicted_label = id2label[np.argmax(predicted_probability)]
    return label2value[predicted_label]

df_t_test["pred"] = df_t_test["text"].apply(predict_label)
df_r_test["pred"] = df_r_test["text"].apply(predict_label)

labels = ['left-leaning', 'center', 'right-leaning']

res_t = evaluate_model(df_t_test['stance'], df_t_test["pred"], labels)
res_r = evaluate_model(df_r_test['stance'], df_r_test["pred"], labels)
display_evaluation_results(res_t, labels, 'tweet-strong-eval.txt')
display_evaluation_results(res_r, labels, 'reddit-strong-eval.txt')