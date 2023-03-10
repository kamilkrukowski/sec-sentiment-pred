from datetime import datetime
from typing import List, Any


import numpy as np


import strategies
from strategies import chronosort
from portfolio_eval import get_strategy_annual_return


START_DATE = '20000101'
END_DATE = '20201231'

strategy = strategies.uniform

labels = []
dates = []
with open('train.tsv') as f:
    f.readline()  # Skip HEADER
    for line in f.readlines():
        _, label, date = line.strip().split('\t')
        dates.append(datetime.strptime(date, '%Y%m%d'))
        labels.append(int(label))

dates, labels = chronosort(dates, labels, START_DATE)

company_list = open('tikrs.txt', 'r').read().strip().split('\n')

allocations = strategy(list(zip(dates, labels)), company_list)

print(get_strategy_annual_return(allocations, company_list,
                                 end_date=END_DATE, start_date=START_DATE))