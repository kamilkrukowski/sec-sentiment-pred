from datetime import datetime
from math import pow

import numpy as np
import matplotlib.pyplot as plt
from pandas import to_datetime

from utils import prog_read_csv
import generate_labels

HOLD_PERIOD = 90
START_YEAR = 2000
START_DATE = datetime.strptime(f'{str(START_YEAR)}-12-31', '%Y-%m-%d')

keys = ['label', 'Date', 'annual return', 'percent return']
dftrain = prog_read_csv('train.tsv', sep='\t', usecols=keys,
                        desc='Loading Training Data')  # load cleaned data

dftrain['Date'] = to_datetime(dftrain['Date'], format="%Y-%m-%d")
dftrain = dftrain.sort_values(by='Date', axis=0)


def evaluate(df):
    # x feature, c class, y price
    c_train = np.array(df['label'])
    ac_train = np.array(df['annual return'])
    pc_train = np.array(df['percent return'])

    mask = c_train != 2
    # ay = ac_train[mask]
    py = pc_train[mask]
    c = c_train[mask]

    # print("\nAbility to distinguish Sentiment Class by using Return as Score"
    #    f"\nAUROC: {roc_auc_score(c, ay):.3f}")
    # print('\nMedian Annualized Return by Sentiment')
    # print(f"Any: {np.median(ay):.3f}")
    # print(f"Negative: {np.median(ay[c==0]):.3f}")
    # print(f"Positive: {np.median(ay[c==1]):.3f}")
    # print('\nMedian Percent Return by Sentiment')
    # print(f"Any: {np.median(py):.3f}")
    # print(f"Negative: {np.median(py[c==0]):.3f}")
    # print(f"Positive: {np.median(py[c==1]):.3f}")

    return np.median(py), np.median(py[c==1])


anys, poss, sps = [], [], []
anyt, post, spst = 1.0, 1.0, 1.0
for year in range(START_YEAR, 2022):
    start = datetime.strptime(f'{year}-01-01', '%Y-%m-%d')
    end = datetime.strptime(f'{year+1}-01-01', '%Y-%m-%d')
    df = dftrain[dftrain.Date > start]
    df = df[df.Date < end]
    any_, pos_ = evaluate(df)

    sp_res = []
    for idx in range(1, 9):
        start2 = datetime.strptime(f'{year}-{idx}-01', '%Y-%m-%d')
        sp_res = generate_labels.n_days_percent_return('^GSPC', start2, 90)
    sp_ = np.mean(sp_res)

    print(f"{year} Alpha_any: {(sp_-any_):.3f}, Alpha_pos: {(pos_-sp_)}:.3f")

    anys.append(any_)
    poss.append(pos_)
    sps.append(sp_)

    anyt = anyt*(1+any_)
    post = post*(1+pos_)
    spst = spst*(1+sp_)

print("Portfolio results")
elapsed_years = 2023 - START_YEAR
exp_ = 4.0/elapsed_years
print(f'anyfiling: {pow(anyt, exp_):.3f}')
print(f'posfiling: {pow(post, exp_):.3f}')
print(f'S&P500: {pow(spst, exp_):.3f}')


plt.plot(anys, label='Any Filing')
plt.plot(poss, label='Positive Filing')
plt.plot(sps, label='S&P500')
plt.savefig('fig.png')
plt.clf();