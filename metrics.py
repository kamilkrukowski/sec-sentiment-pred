from collections import UserDict
from math import ceil

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, roc_curve, precision_recall_curve


class Metrics(UserDict):

    def __init__(self):
        super().__init__()

    def calculate(self, y, y_hat, split=''):
        class_ = 1
        y = np.array(y) == class_
        y_hat = np.array(y_hat)[:, class_]
        self[f"{split}_auroc"] = roc_auc_score(y, y_hat)
        self[f"_{split}_acc_pos"] = accuracy_score(y, y_hat > 0.5)
        self[f"_{split}_%pos"] = sum(y)/len(y)
        self[f"{split}_acc_over_random"] = self[f"_{split}_acc_pos"] - self[f"_{split}_%pos"]
        self[f"{split}_Pat10%"] = Metrics.PatK(y, y_hat, K=0.1)
        self[f"_{split}_ROC"] = Metrics.roc_curve(y, y_hat)
        self[f"_{split}_PRC"] = Metrics.pr_curve(y, y_hat)

    @staticmethod
    def roc_curve(labels, predictions):
        fpr, tpr, _ = roc_curve(labels, predictions)
        return (fpr, tpr)

    @staticmethod
    def pr_curve(labels, predictions):
        prec, rec, _ = precision_recall_curve(labels, predictions)
        return (rec, prec)

    @staticmethod
    def acc(labels, predictions):
        return sum(labels == (predictions > 0.5))/sum(labels)

    @staticmethod
    def PatK(labels, predictions, K='10%', average='best'):
        if K < 1.0:
            K = ceil(len(labels) * K)

        assert isinstance(K, int)

        idxs_top = np.argsort(predictions)[::-1][:K]
        labels = labels[idxs_top]

        return labels.mean()

    def __str__(self):
        return "{ " + ", ".join(
            [f"{key}: {self[key]:.3f}" for key in self if key[0] != '_']) + " }"

    def __repr__(self):
        return str(self)
