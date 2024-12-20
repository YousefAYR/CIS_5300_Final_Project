# -*- coding: utf-8 -*-
"""Georgetown_Labeling.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/17tcVj2QAWSP_TsIa-l7HRywo40Toe7Ei
"""

!pip install pandas transformers scikit-learn
!pip install kaggle



from google.colab import drive
drive.mount('/content/CIS5300_Project')

!kaggle datasets download -d manchunhui/us-election-2020-tweets -p ./

Aimport zipfile
with zipfile.ZipFile('./us-election-2020-tweets.zip', 'r') as zip_ref:
    zip_ref.extractall('./')

import os
import pandas as pd
# List files in the target directory
folder_path = '.'
files = os.listdir(folder_path)
print(files)

biden_data = pd.read_csv('./hashtag_donaldtrump.csv', lineterminator='\n')
trump_data = pd.read_csv('./hashtag_joebiden.csv', lineterminator='\n')


# Check the info of the datasets
print("Joe Biden Dataset Info:")
print(biden_data.info())
print("\nDonald Trump Dataset Info:")
print(trump_data.info())

# Display a preview of each dataset
print("\nJoe Biden Dataset Preview:")
print(biden_data.head())
print("\nDonald Trump Dataset Preview:")
print(trump_data.head())

# Getting biden dataset information
biden_data.info()
trump_data.info()

trump_data['candidate'] = 'trump'

# biden dataframe
biden_data['candidate'] = 'biden'

# combining the dataframes
data = pd.concat([trump_data, biden_data])

# FInal data shape
print('Final Data Shape :', data.shape)

# View the first 2 rows
print("\nFirst 2 rows:")
print(data.head(3))

# List of columns to remove
columns_to_remove = [
    'user_followers_count', 'source', 'user_id',
    'user_name', 'user_screen_name',
    'collected_at', 'likes', 'retweet_count'
]

# Drop the columns
data = data.drop(columns=columns_to_remove)

# dropping null values if they exist
data.dropna(inplace=True)

data['country'].value_counts()

data['country'] = data['country'].replace({'United States of America': "US",'United States': "US"})
data.info()

import matplotlib.pyplot as plt
import plotly.express as px
import plotly.io as pio
pio.renderers.default = "vscode"
# Group the data by 'candidate' and count the
# number of tweets for each candidate
tweets_count = data.groupby('candidate')['tweet'].count().reset_index()

# Interactive bar chart
fig = px.bar(tweets_count, x='candidate', y='tweet', color='candidate',
color_discrete_map={'Trump': 'pink', 'Biden': 'blue'},
labels={'candidate': 'Candidates', 'tweet': 'Number of Tweets'},
title='Tweets for Candidates')

# Show the chart
fig.show()

data['country'] = data['country'].replace({'United States of America': "US",'United States': "US"})

# Group the data by 'candidate' and count the
# number of tweets for each candidate
tweets_count = data.groupby('candidate')['tweet'].count().reset_index()

# Interactive bar chart
fig = px.bar(tweets_count, x='candidate', y='tweet', color='candidate',
color_discrete_map={'Trump': 'pink', 'Biden': 'blue'},
labels={'candidate': 'Candidates', 'tweet': 'Number of Tweets'},
title='Tweets for Candidates')

# Show the chart
fig.show()

import re
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
def clean(text):
	# Remove URLs
	text = re.sub(r'https?://\S+|www\.\S+', '', str(text))

	# Convert text to lowercase
	text = text.lower()

	# Replace anything other than alphabets a-z with a space
	text = re.sub('[^a-z]', ' ', text)

	# Split the text into single words
	text = text.split()

	# Initialize WordNetLemmatizer
	lm = WordNetLemmatizer()

	# Lemmatize words and remove stopwords
	text = [lm.lemmatize(word) for word in text if word not in set(
		stopwords.words('english'))]

	# Join the words back into a sentence
	text = ' '.join(word for word in text)

	return text

!pip install transformers
!pip install torch

import re
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import numpy as np

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using {device}')
biden_classifier = 'kornosk/bert-election2020-twitter-stance-biden-KE-MLM'
tokenizer_b = AutoTokenizer.from_pretrained(biden_classifier)
model_b = AutoModelForSequenceClassification.from_pretrained(biden_classifier)

trump_classifier = 'kornosk/bert-election2020-twitter-stance-trump-KE-MLM'
tokenizer_t = AutoTokenizer.from_pretrained(trump_classifier)
model_t = AutoModelForSequenceClassification.from_pretrained(trump_classifier)

classifier = {
    'biden': {
        'tokenizer' : tokenizer_b,
        'model' : model_b,
        'multiplier' : -1 #adjust to relative to trump
    },
    'trump': {
        'tokenizer' : tokenizer_t,
        'model' : model_t,
        'multiplier' : 1 #adjust to relative to trump
    }
}

id2label = {
    0: "AGAINST",
    1: "FAVOR",
    2: "NONE"
}
label2id = {
    "AGAINST": -1, # left-leaning
    "FAVOR": 1, # right-leaning
    "NONE": 0 # neutral
}
id2newlbl = {
    0: "neutral",
    1: "pro-Trump",
    -1: "pro-Biden"
}

def classify(row):
    sentence = row['tweet']
    candidate = row['candidate']
    inputs = classifier[candidate]['tokenizer'](sentence.lower(), return_tensors='pt').to(device)
    model = classifier[candidate]['model'].to(device)
    outputs = model(**inputs)
    predicted_probability = torch.softmax(outputs[0], dim=1)[0].tolist()
    prediction = id2label[np.argmax(predicted_probability)]
    id = classifier[candidate]['multiplier'] * label2id[prediction]
    text_prediction = id2newlbl[id]
    return pd.Series([id, text_prediction])

def clean(text):
    # Remove URLs
    text = re.sub(r'https?://\S+|www\.\S+', '', str(text))

    # Convert text to lowercase
    text = text.lower()

    # Replace anything other than alphabets a-z with a space
    text = re.sub('[^a-z]', ' ', text)

    # Split the text into single words
    text = text.split()

    # Initialize WordNetLemmatizer
    lm = WordNetLemmatizer()

    # Lemmatize words and remove stopwords
    text = [lm.lemmatize(word) for word in text if word not in set(
        stopwords.words('english'))]

    # Join the words back into a sentence
    text = ' '.join(word for word in text)

    return text

import swifter
data['tweet'] = data['tweet'].swifter.apply(clean)
data = data.reset_index(drop=True)

data[['stance', 'label']] = data.swifter.apply(classify, axis=1)

data.to_csv('labeled_tweets.csv', index=False)

print(data['label'].value_counts())

import matplotlib.pyplot as plt

data['label'].value_counts().plot(kind='bar', title='Tweet Sentiment Distribution')
plt.xlabel('Sentiment')
plt.ylabel('Count')
plt.show()