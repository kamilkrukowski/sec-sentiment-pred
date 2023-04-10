import torch.nn as nn
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
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


def GET_FFNBOW_RESULTS(dftrain, dfval, dftest, threshold=0.9, out_inplace=False,
                       pbar=None, pbar_desc=None):

    pbar.set_description(desc=f"{pbar_desc}: Fitting Embedder")
    vectorizer = Pipeline(
       [('tfidf', TfidfVectorizer(min_df=20, max_df=0.8, max_features=20000))])
    vectorizer.fit(dftrain['text'])

    pbar.set_description(desc=f"{pbar_desc}: Loading Model")
    shape = vectorizer.transform(dftest['text'].iloc[0:3]).shape[1]
    model = BOW(shape)
    optim = torch.optim.Adam(model.parameters(), weight_decay=1e-2)

    train_dataloader = DataLoader(
                        MyDataset(dftrain), batch_size=512, shuffle=True)
    val_dataloader = DataLoader(
                        MyDataset(dfval), batch_size=512,
                        shuffle=True)
    test_dataloader = DataLoader(
                        MyDataset(dftest), batch_size=len(dftest),
                        shuffle=True)

    x_test, y_test = next(iter(test_dataloader))
    x_test = torch.tensor(
                    vectorizer.transform(x_test).todense()).float()

    metrics = Metrics()

    len_train = len(train_dataloader)
    for epoch in range(N_EPOCHS):
        for idx, (x_train, y_train) in enumerate(train_dataloader):
            pbar.set_description(
               desc=f"{pbar_desc}: Epoch {epoch}: Batch {idx+1}/{len_train}")
            x_train = torch.tensor(
                        vectorizer.transform(x_train).todense()).float()
            y_hat = model(x_train)

            loss = torch.nn.functional.cross_entropy(y_hat, y_train)

            optim.zero_grad()
            loss.backward()
            optim.step()

        with torch.no_grad():
            loss_val = 0
            for x_val, y_val in val_dataloader:
                x_val = torch.tensor(
                            vectorizer.transform(x_val).todense()).float()
                t = model(x_val)
                loss_val += torch.nn.functional.cross_entropy(
                                    t, y_val, reduction='mean').detach().numpy()
            loss_val = loss_val / len(val_dataloader)

        met_ = loss_val
        pbar.set_description(desc=f"{pbar_desc}: Loss={met_:.3f}:")
    metrics['loss_val'] = loss_val
    metrics['train_loss'] = loss.detach().numpy().item()

    yhat_test = model(x_test).detach().numpy()
    yhat_train = model(x_train).detach().numpy()

    metrics.calculate(y_test, yhat_test, split='test')
    metrics.calculate(y_train, yhat_train, split='train')

    scores = model(x_test)
    pos_scores = scores[:, 1].detach().numpy()
    preds = pos_scores > np.percentile(pos_scores, int(threshold*100))

    out = dftest
    if not out_inplace:
        out = out.copy()

    out['pred'] = preds
    out['score'] = pos_scores

    if not out_inplace:
        out.drop('text', axis=1)

    return out, metrics
