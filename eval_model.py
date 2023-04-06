import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.model_selection import KFold
import seaborn as sns 
from collections import defaultdict
import numpy as np
import os


from utils import prog_read_csv, generate_data_splits
from dataloading import HistoricalYahoo, calculate_metrics, get_reference_data
from bow import GET_BOW_RESULTS
from model import GET_FFNBOW_RESULTS

INPUT_DATA_NAME = '8K_data_short.tsv'
# INPUT_DATA_NAME = '8K_data.tsv'
PICKLED_YFINANCE = 'TIKR_DATA.pkl'
HOLD_PERIOD=90

EVAL_YEAR = 2018
EVAL_YEAR = datetime.strptime(str(EVAL_YEAR) + '0101', '%Y%m%d')
model_savepath = 'test1'

K = 3
# Load data
keys = ['text', 'label', 'Date', 'tikr']
data = prog_read_csv(INPUT_DATA_NAME, sep='\t',
                     usecols=keys, desc='Loading Data')
data['Date'] = pd.to_datetime(data.Date, format='%Y-%m-%d')

# Load historical returns
yd = HistoricalYahoo()
yd.preload_tikrs(data)
calculate_metrics(yd, HOLD_PERIOD, silent=True)

strategy = 'chronological_yearly'
model = GET_BOW_RESULTS
# model = GET_FFNBOW_RESULTS
all_metrics = []
output_df = pd.DataFrame()
N_YEARS = 2023-EVAL_YEAR.year+1
for idx, (train_val_df, testdf) in enumerate(
            generate_data_splits(data, strategy=strategy,
                                 periods_to_test=N_YEARS)):

    # x_train, y_train = traindf['text'], traindf['label']
    # x_test, y_test = testdf['text'], testdf['label']
    
    for k, (train_index, test_index) in enumerate(KFold(n_splits=K, shuffle=True).split(train_val_df)):
        traindf = train_val_df.iloc[train_index]
        validationdf = train_val_df.iloc[test_index]

        """
        total_test = len(y_test)
        total_train = len(y_train)
        for idx, label in enumerate(['negative', 'positive', 'neutral']):
            prop_train = sum(y_train == idx)/total_train
            prop_test = sum(y_test == idx)/total_test
            print(f"{label} - test: {prop_test:0.4f}, train: {prop_train:0.4f}")
        """

        out, metrics = model(traindf, validationdf, testdf)
        out['Date'] = pd.to_datetime(out['Date'], format="%Y%m%d")
        metrics.year = testdf.Date.iloc[0].year
        metrics.k = k + 1
        print(f'{metrics.year}, k = {k+1}')
        print(metrics)
        all_metrics.append(metrics)

        get_reference_data(out, yd, cols=['Annual Return', 'beta', 'sp Annual', 'sp Percent'])

        temp_df = pd.DataFrame()
        temp_df[['label', 'pred', 'score']] = out[['label', 'pred', 'score']]
        temp_df['pred'] = temp_df['pred'].astype(int)
        temp_df['year'] = [metrics.year]*len(out)
        temp_df['k'] = [k+1]*len(out)
        output_df = pd.concat([output_df, temp_df])

output_df.reset_index(drop=True).to_csv(f'model_outputs/outputs_{model_savepath}.csv')
"""
out = out[out.beta != -999]

any_adv_ = (out['Annual Return'] - out['sp Annual']).mean()
any_alph = ((out['Annual Return'] - out['sp Annual'])/out['beta']).mean()

out_ = out[out.pred == 1]
pos_adv_ = (out_['Annual Return'] - out_['sp Annual']).mean()
pos_alph = ((out_['Annual Return'] - out_['sp Annual'])/out_['beta']).mean()

print(f"Any: Return: {out['Annual Return'].mean():.2f}, S&P500: {out['sp Annual'].mean():.2f}"
    f" Beta: {out['beta'].mean():.2f}")
print(f"Pos: Return: {out_['Annual Return'].mean():.2f}, S&P500: {out_['sp Annual'].mean():.2f}"
    f" Beta: {out_['beta'].mean():.2f}")

sharpe_sp = out['sp Annual'].mean()/out['sp Annual'].std() 
sharpe_any = out['Annual Return'].mean()/out['Annual Return'].std() 
sharpe_pos = out_['Annual Return'].mean()/out_['Annual Return'].std() 
print(f"Sharpe Ratios: S&P500: {sharpe_sp:.2f} Any: {sharpe_any:.2f} Pos: {sharpe_pos:.2f}")

cols = ['Date', 'label', 'pred', 'score', 'tikr']
out[cols].to_csv('results_bow.tsv', sep='\t')
"""""

