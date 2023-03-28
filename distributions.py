from datetime import datetime
from math import pow

import numpy as np
import matplotlib.pyplot as plt
from pandas import to_datetime

from utils import prog_read_csv
import generate_labels as gl

HOLD_PERIOD = 90
START_YEAR = 2000
START_DATE = datetime.strptime(f'{str(START_YEAR)}-01-01', '%Y-%m-%d')

keys = ['label', 'Date', 'tikr']
dftrain = prog_read_csv('8k_data.tsv', sep='\t', usecols=keys,
                        desc='Loading Training Data')  # load cleaned data

dftrain['Date'] = to_datetime(dftrain['Date'], format="%Y-%m-%d")
dftrain = dftrain.sort_values(by='Date', axis=0)

dftrain = dftrain.rename(columns={'label': 'orginal_label', 'pred': 'label'})


def evaluate(df):
    # x feature, c class, y price
    c_train = np.array(df['label'])
    pc_train = np.array(df['percent return'])

    mask = c_train != 2
    py = pc_train[mask]
    c = c_train[mask]

    return np.median(py), np.median(py[c==1])

"""
def get_group_beta(tikr_list):
    tikrs, counts = np.unique(tikr_list, return_counts=True)
    weights = counts/counts.sum()
    tikr_betas = np.array([calculator.get_single_beta(i) for i in tikrs])
    tikr_betas = tikr_betas.flatten()
    beta = (tikr_betas * weights).sum()
    return beta
"""

anys, poss, sps, betas, pbetas = [], [], [], [], []
aanys, aposs = [], []
anyt, post, spst = 1.0, 1.0, 1.0
num_years = 2022 - START_YEAR
for year in range(START_YEAR, 2022):
    start = datetime.strptime(f'{year}-01-01', '%Y-%m-%d')
    end = datetime.strptime(f'{year+1}-01-01', '%Y-%m-%d')
    df = dftrain[dftrain.Date > start]
    df = df[df.Date < end]
    #any_, pos_ = evaluate(df)

    #beta_ = list(df.tikr)
    #betap_ = list(df[df.label == 1].tikr)

    sp_res = []
    for idx in range(1, 9):
        start2 = datetime.strptime(f'{year}-{idx}-01', '%Y-%m-%d')
        sp_res = gl.n_days_percent_return('^GSPC', start2, 90)
    sp_ = np.mean(sp_res)

    """
    beta_any = get_group_beta(beta_)
    beta_pos = get_group_beta(betap_)

    a_any = any_ - sp_*beta_any
    a_pos = pos_ - sp_*beta_pos

    print(f"{year} Alpha_any: {a_any:.3f}, Alpha_pos: {a_pos:.3f}")
    """

    #anys.append(any_)
    #poss.append(pos_)
    sps.append(sp_)
    #betas.append(beta_)
    #pbetas.append(betap_)

    #aanys.append(a_any)
    #aposs.append(a_pos)

    #anyt = anyt*(1+any_)
    #post = post*(1+pos_)
    spst = spst*(1+sp_)

aanys = np.array(aanys)
aposs = np.array(aposs)

t = 1
for i in aanys:
    t = t*(1+i)
t = pow(t, 1.0/num_years) - 1 # Back-annualize from long period
print(f"Calculated (Any) Alpha: {t:.3f}")

t = 1
for i in aposs:
    t = t*(1+i)
t = pow(t, 1.0/num_years) - 1 # Back-annualize from long period
print(f"Calculated (POS) Alpha: {t:.3f}")

"""
betas = [i for j in betas for i in j]
pbetas = [i for j in pbetas for i in j]

tikrs, counts = np.unique(betas, return_counts=True)
weights = counts/counts.sum()
tikr_betas = np.array([calculator.get_single_beta(i) for i in tikrs])
tikr_betas = tikr_betas.flatten()
beta = (tikr_betas * weights).sum()
print(f"Overall ANY Beta: {beta:.3f}")

tikrs, counts = np.unique(pbetas, return_counts=True)
weights = counts/counts.sum()
tikr_betas = np.array([calculator.get_single_beta(i) for i in tikrs])
tikr_betas = tikr_betas.flatten()
beta = (tikr_betas * weights).sum()
print(f"Overall POS Beta: {beta:.3f}")

print("Portfolio results")
elapsed_years = 2023 - START_YEAR
exp_ = (365.0/HOLD_PERIOD)/elapsed_years
print(f'anyfiling: {pow(anyt, exp_):.3f}')
print(f'posfiling: {pow(post, exp_):.3f}')
print(f'S&P500: {pow(spst, exp_):.3f}')
"""

"""
plt.plot(anys, label='Any Filing')
plt.plot(poss, label='Positive Filing')
plt.plot(sps, label='S&P500')
plt.savefig('fig.png')
plt.clf();
"""