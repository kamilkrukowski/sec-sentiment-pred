import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

# # vectorize
dftrain = pd.read_csv('train.tsv', sep='\t')  # load cleaned data
dftest = pd.read_csv('test.tsv', sep='\t')  # load cleaned data

vectorizer = TfidfVectorizer()

X_train, y_train = dftrain['text'], dftrain['label']
X_test, y_test = dftest['text'], dftest['label']

X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

clf = LogisticRegression(random_state=1).fit(X_train, y_train)

total_test = len(y_test)
total_train = len(y_train)
for idx, label in enumerate(['negative', 'positive', 'neutral']):
    prop_train = sum(y_train == idx)/total_train
    prop_test = sum(y_test == idx)/total_test
    print(f"{label} - test: {prop_test:0.4f}, train: {prop_train:0.4f}")


def auroc(labels, predictions, average='best'):
    aucs = []
    for i in range(3):
        aucs.append(roc_auc_score(labels == i, predictions[:, i]))

    if average == 'macro':
        return sum(aucs)/3
    if average == 'best':
        return aucs[1]
    if average is None:
        return aucs


def acc(labels, predictions, average='macro'):
    accs = []
    for i in range(3):
        idxs = labels == i
        accs.append(sum(labels[idxs] == np.argmax(
            predictions[idxs], axis=1))/sum(idxs))

    if average == 'macro':
        return sum(accs)/3
    if average == 'best':
        return accs[1]
    if average is None:
        return accs


print(f"test auroc: {auroc(y_test, clf.predict_proba(X_test)):.3f}")
print(f"train auroc: {auroc(y_train, clf.predict_proba(X_train)):.3f}")
print(f"test acc: {acc(y_test, clf.predict_proba(X_test)):.3f}")
print(f"train acc: {acc(y_train, clf.predict_proba(X_train)):.3f}", )
