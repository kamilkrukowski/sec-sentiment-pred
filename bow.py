import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from metrics import Metrics
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
def GET_BOW_RESULTS(dftrain, dfval, dftest, threshold=0.9, out_inplace=False):

    x_train, y_train = dftrain['text'], dftrain['label']
    x_test, y_test = dftest['text'], dftest['label']
    x_val, y_val = dfval['text'], dfval['label']

    vectorizer = TfidfVectorizer(min_df=20, max_df=0.8)

    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_val = vectorizer.transform(x_val)

    clf = LogisticRegression(C=1.0, random_state=1).fit(x_train, y_train)

    yhat_test = clf.predict_proba(x_test)
    yhat_val = clf.predict_proba(x_val)
    yhat_train = clf.predict_proba(x_train)

    metrics = Metrics()
    metrics['Train features'] = int(x_train.shape[1])
    metrics.calculate(y_test, yhat_test, split='test')
    metrics.calculate(y_val, yhat_val, split='validation')
    metrics.calculate(y_train, yhat_train, split='train')
    
    scores = clf.predict_proba(x_test)
    pos_scores = scores[:, 1]
    preds = pos_scores > np.percentile(pos_scores, int(threshold*100))

    out = dftest
    if not out_inplace:
        out = out.copy()

    out['pred'] = preds
    out['score'] = pos_scores

    if not out_inplace:
        out.drop('text', axis=1)

    return out, metrics


def GET_DECISION_TREE_RESULTS(dftrain, dfval, dftest, max_depth=2, threshold=0.9,out_inplace=False, is_decision_tree = True):

    x_train, y_train = dftrain['text'], dftrain['label']
    x_test, y_test = dftest['text'], dftest['label']
    x_val, y_val = dfval['text'], dfval['label']

    vectorizer = TfidfVectorizer(min_df=20, max_df=0.8)

    x_train = vectorizer.fit_transform(x_train)
    x_test = vectorizer.transform(x_test)
    x_val = vectorizer.transform(x_val)

    clf = RandomForestClassifier(max_depth= max_depth, random_state=0).fit(x_train, y_train)
    if(is_decision_tree == True):
        clf = DecisionTreeClassifier(max_depth= max_depth,random_state=0).fit(x_train, y_train)
        

    yhat_test = clf.predict_proba(x_test)
    yhat_val = clf.predict_proba(x_val)
    yhat_train = clf.predict_proba(x_train)

    metrics = Metrics()
    metrics['Train features'] = int(x_train.shape[1])
    metrics.calculate(y_test, yhat_test, split='test')
    metrics.calculate(y_val, yhat_val, split='validation')
    metrics.calculate(y_train, yhat_train, split='train')
    
    scores = clf.predict_proba(x_test)
    pos_scores = scores[:, 1]
    preds = pos_scores > np.percentile(pos_scores, int(threshold*100))

    out = dftest
    if not out_inplace:
        out = out.copy()

    out['pred'] = preds
    out['score'] = pos_scores

    if not out_inplace:
        out.drop('text', axis=1)

    return out, metrics
