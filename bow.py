import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# # vectorize
df = pd.read_csv('train.csv') # load cleaned data
vectorizer = TfidfVectorizer()
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['label'], test_size=0.25, random_state=42)
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

clf = LogisticRegression(random_state=1).fit(X_train, y_train)
print("test score: ", clf.score(X_test, y_test))
print("train score: ", clf.score(X_train, y_train))