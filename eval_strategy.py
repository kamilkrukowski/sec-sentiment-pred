from datetime import datetime
from typing import List, Any


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

def chronosort(dates: List[datetime], arr: List[Any],
               start_date: datetime, end_date: datetime):
    """Sort dates and arr in-place by dates, remove all
    entries before start_date or after end_date"""

    if type(start_date) is str:
        start_date = datetime.strptime(start_date, '%Y%m%d') 

    if type(end_date) is str:
        end_date = datetime.strptime(end_date, '%Y%m%d') 

    dates = np.array(dates)
    arr = np.array(arr)

    sort_idxs = np.argsort(dates)
    arr = arr[sort_idxs]
    dates = dates[sort_idxs]

    dates = list(dates)
    dates.insert(0, start_date)
    arr = list(arr)
    arr.insert(0, -999)
    
    if end_date is not None:
        dates.insert(1, end_date)
        arr.insert(1, -999)

    sort_idxs = list(np.argsort(dates))
    start_idx = sort_idxs.index(0) + 1  # We exclude our dummy entry
    end_idx = None

    if end_date is not None:
        end_idx = sort_idxs.index(1)

    dates = np.array(dates)
    arr = np.array(arr)
    dates = dates[sort_idxs][start_idx:end_idx]
    arr = arr[sort_idxs][start_idx:end_idx]

    return dates, arr


dates, labels = chronosort(dates, labels, START_DATE)

company_list = open('tikrs.txt', 'r').read().strip().split('\n')

allocations = strategy(labels, company_list)

print(get_strategy_annual_return(allocations, company_list, end_date='20220101'))