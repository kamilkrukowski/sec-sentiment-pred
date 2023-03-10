from datetime import datetime


import numpy as np


import strategies
from portfolio_eval import get_strategy_annual_return


START_DATE = datetime.strptime('20000101', '%Y%m%d')

strategy = strategies.baseline

labels = []
dates = []
with open('train.tsv') as f:
    f.readline()  # Skip HEADER
    for line in f.readlines():
        _, label, date = line.strip().split('\t')
        dates.append(datetime.strptime(date, '%Y%m%d'))
        labels.append(int(label))

def chronosort(dates, arr, start_date: datetime):
    dates = np.array(dates)
    labels = np.array(labels)

    sort_idxs = np.argsort(dates)
    labels = labels[sort_idxs]
    dates = dates[sort_idxs]

    dates = list(dates)
    dates.insert(0, START_DATE)
    labels = list(labels)
    labels.insert(0, -999)

    sort_idxs = list(np.argsort(dates))
    found_idx = sort_idxs.index(0) + 1  # We exclude our dummy entry

    dates = np.array(dates)
    labels = np.array(labels)
    dates = dates[sort_idxs][found_idx:]
    labels = labels[sort_idxs][found_idx:]

    return dates, labels


dates, labels = chronosort(dates, labels, START_DATE)

company_list = open('tikrs.txt', 'r').read().strip().split('\n')

allocations = strategy(labels, company_list)

print(get_strategy_annual_return(allocations, company_list))