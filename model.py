import torch.nn as nn
import numpy as np
import torch

N_EPOCHS = 50

class BOW(torch.nn.Module):

    def __init__(self, input_size: int):
        super(BOW, self).__init__()

        INPUT_SIZE = input_size

        self.ffn = torch.nn.Sequential(
            nn.Linear(INPUT_SIZE, 4096),
            nn.ReLU(),
            nn.Linear(4096, 3),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        return self.ffn(x)

from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline

# # vectorize
df = pd.read_csv('train.csv') # load cleaned data

LM_dict = pd.read_csv('LM_dict.csv',keep_default_na=False)
#create vocab from LM dict
vocab = dict()
for index, row in LM_dict.iterrows():
    word = row['Word'].lower()
    vocab[word] = index
pipe = Pipeline([('count', CountVectorizer(vocabulary=vocab)),
                 ('tfid', TfidfTransformer())]).fit(df['text'])# tfidf on 86000 vocab
X = pipe.transform(df['text'])
sort_index = np.argsort(X.toarray().sum(axis = 0))# ascending
new_vocab = pipe['count'].get_feature_names_out()[sort_index[-2000:]]#take top 500 vocab

vectorizer = TfidfVectorizer() #pass new_vocab if desired
x_train, x_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)
x_train = vectorizer.fit_transform(x_train)
x_test = vectorizer.transform(x_test)

x_train = torch.Tensor(x_train.todense())
y_train = torch.Tensor(np.array(y_train)).long()
x_test = torch.Tensor(x_test.todense())
y_test = torch.Tensor(np.array(y_test)).long()

model = BOW(x_train.shape[1])
optim = torch.optim.Adam(model.parameters(), weight_decay=2e-3)

def accuracy(logits, y):
    y_hat = torch.argmax(logits, dim=1)
    return (sum(y_hat==y)/len(y)).detach().item()

for epoch in range(N_EPOCHS):
    y_hat = model(x_train)

    loss = torch.nn.functional.cross_entropy(y_hat, y_train)

    optim.zero_grad()
    loss.backward()
    optim.step()
    print(f"Loss: {loss.detach().item():.2f}, \
            Train Acc: {accuracy(y_hat, y_train)*100:.2f} \
            Test Acc: {accuracy(model(x_test), y_test)*100:.2f}")