import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset
import evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SentimentClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, dropout=0.5):
        super(SentimentClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=2,
            batch_first=True,
            bidirectional=True,
        )
        self.fc1 = nn.Linear(hidden_dim * 2, 64)
        self.fc2 = nn.Linear(64, 16)
        self.fc3 = nn.Linear(16, num_classes)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        embedded = self.embedding(x)
        lstm_out, _ = self.lstm(embedded)
        avg_pool = torch.mean(lstm_out, dim=1)
        x = F.relu(self.fc1(avg_pool))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        logits = self.fc3(x)
        return logits

class TextDataset(Dataset):
    def __init__(self, dataframe, tokenizer, vocab):
        self.texts = dataframe['text'].tolist()
        self.labels = dataframe['stance'].tolist()
        self.tokenizer = tokenizer
        self.vocab = vocab

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        tokenized_text = torch.tensor(self.vocab(self.tokenizer(text)), dtype=torch.long)
        return tokenized_text, label

tokenizer = get_tokenizer("basic_english")

tweet_path = 'data/labeled_tweets_georgetown/'
reddit_path = 'data/factoid_reddit/'

df_train = pd.read_csv(reddit_path + 'train.csv', on_bad_lines='skip')
df_dev = pd.read_csv(reddit_path + 'dev.csv', on_bad_lines='skip')
df_test = pd.read_csv(reddit_path + 'test.csv', on_bad_lines='skip')

def yield_tokens(data_iter):
    for text in data_iter:
        yield tokenizer(text)

vocab = build_vocab_from_iterator(yield_tokens(df_train['text'].tolist()), specials=["<unk>", "<pad>"])
vocab.set_default_index(vocab["<unk>"])

vocab_size = len(vocab)
embedding_dim = 128
hidden_dim = 128
num_classes = 3
max_length = 512
learning_rates = [0.01]

def text_pipeline(text):
    tokenized = vocab(tokenizer(text))
    if len(tokenized) > max_length:
        tokenized = tokenized[:max_length]
    return tokenized

def collate_fn(batch):
    texts, labels = zip(*batch)
    texts = [torch.tensor(text_pipeline(text), dtype=torch.long) for text in texts]
    padded_texts = pad_sequence(texts, batch_first=True, padding_value=vocab["<pad>"])
    labels = torch.tensor(labels, dtype=torch.long)
    return padded_texts, labels



train_dataset = TextDataset(df_train, tokenizer, vocab)
dev_dataset = TextDataset(df_dev, tokenizer, vocab)
test_dataset = TextDataset(df_test, tokenizer, vocab)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=collate_fn, shuffle=True)
dev_loader = DataLoader(dev_dataset, batch_size=batch_size, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, collate_fn=collate_fn)

model = SentimentClassifier(vocab_size, embedding_dim, hidden_dim, num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

def evaluate_model(model, data_loader, criterion):
    model.eval()
    total_loss = 0
    with torch.no_grad():
        for batch_texts, batch_labels in data_loader:
            outputs = model(batch_texts)
            loss = criterion(outputs, batch_labels)
            total_loss += loss.item()
    return total_loss / len(data_loader)

def train_model(model, train_loader, dev_loader, learning_rates, num_epochs=10):
    best_loss = float('inf')
    best_lr = None

    for lr in learning_rates:
        print(f"Training with learning rate: {lr}")
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0
            for batch_texts, batch_labels in train_loader:
                optimizer.zero_grad()
                outputs = model(batch_texts)
                loss = criterion(outputs, batch_labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            print(f"Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss/len(train_loader)}")

        dev_loss = evaluate_model(model, dev_loader, criterion)
        print(f"Dev Loss for learning rate {lr}: {dev_loss}")

        if dev_loss < best_loss:
            best_loss = dev_loss
            best_lr = lr
            torch.save(model.state_dict(), "best_model.pth")

    print(f"Best learning rate: {best_lr}")
    return best_lr


model.load_state_dict(torch.load("best_model.pth"))
def predict_on_test(model, data_loader):
    model.eval()
    predictions = []
    true_labels = []
    with torch.no_grad():
        for batch_texts, batch_labels in data_loader:
            outputs = model(batch_texts)
            predicted_labels = torch.argmax(outputs, dim=1)
            predictions.extend(predicted_labels.tolist())
            true_labels.extend(batch_labels.tolist())
    return true_labels, predictions

y_true, y_pred = predict_on_test(model, test_loader)

labels = ['left-leaning', 'center', 'right-leaning']
results = evaluate.evaluate_model(y_true, y_pred, labels)
evaluate.display_evaluation_results(results, labels, 'reddit-strong-2-eval.txt')


