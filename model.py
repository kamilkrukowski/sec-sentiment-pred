import torch.nn as nn
from torch.utils.data import Dataset
import torch

class Dataset_8K(Dataset):

    def __init__(self, split='train'):

        self.x = []
        self.y = []

        if split == 'train':
            fpath = 'train.csv'
            self._load_file_(fpath);

        if split == 'test':
            fpath = 'test.csv'
            self._load_file_(fpath);

    def _load_file_(self, fpath):
        with open(fpath, 'r') as f:
            _ = f.readline() # Header to delete
            for line in f.readlines():
                curr_x, curr_y, *r = tuple(line.split(','))
                self.x.append(curr_x)
                self.y.append(int(curr_y))

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

data = Dataset_8K()

import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
import re
from tqdm import tqdm
from time import time

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
new_vocab = pipe['count'].get_feature_names_out()[sort_index[-500:]]#take top 500 vocab

#############################################
# # training with new vocab list
vectorizer = TfidfVectorizer()
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)
from sklearn.linear_model import LogisticRegression
clf = LogisticRegression(random_state=1).fit(X_train, y_train)
print("test score: ", clf.score(X_test, y_test))
print("train score: ",clf.score(X_train, y_train))