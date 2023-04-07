import torch.nn as nn
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from metrics import Metrics

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

class MyDataset(Dataset):

  def __init__(self, df):

    self.x = df['text']
    self.y = torch.tensor(df['label'].values, dtype=torch.long) 

  def __len__(self):
    return len(self.y)

  def __getitem__(self, idx):
    return self.x.iloc[idx], self.y[idx]


def GET_FFNBOW_RESULTS(dftrain, dfval, dftest, threshold=0.9, out_inplace=False):

    print("Fitting vectorizer")
    vectorizer = TfidfVectorizer(min_df=20, max_df=0.8, max_features=20000)  # pass new_vocab if desired
    vectorizer.fit(dftrain['text'])
    print("vectorizer fit")

    model = BOW(vectorizer.transform(dftest['text'].iloc[0:3]).shape[1])
    optim = torch.optim.Adam(model.parameters(), weight_decay=2e-3)

    train_dataloader = DataLoader(
                        MyDataset(dftrain), batch_size=512, shuffle=True)
    val_dataloader = DataLoader(
                        MyDataset(dfval), batch_size=512,
                        shuffle=True)
    test_dataloader = DataLoader(
                        MyDataset(dftest), batch_size=len(dftest),
                        shuffle=True)

    x_test, y_test = next(iter(test_dataloader))
    x_test = torch.tensor(vectorizer.transform(x_test).todense()).float()

    metrics = Metrics()

    for epoch in range(N_EPOCHS):
        print(f"Epoch: {epoch}")
        for x_train, y_train in tqdm(train_dataloader, desc='Epoching'):
            x_train = torch.tensor(
                        vectorizer.transform(x_train).todense()).float()
            y_hat = model(x_train)

            loss = torch.nn.functional.cross_entropy(y_hat, y_train)

            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            for x_val, y_val in tqdm(train_dataloader, desc='Epoching'):
                x_val = torch.tensor(
                            vectorizer.transform(x_val).todense()).float()
                t = model(x_val)
                class_ = 1
                y = np.array(y_test) == class_
                y_hat = np.array(t)[:, class_]
                auroc = roc_auc_score(y, y_hat)
                print(f"val auroc: {auroc:.3f}")

    yhat_test = model(x_test)
    yhat_train = model(x_train)

    metrics.calculate(y_test, yhat_test, split='test')
    metrics.calculate(y_train, yhat_train, split='train')

    scores = model(x_test)
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
