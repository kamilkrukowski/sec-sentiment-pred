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
from bow import GET_BOW_RESULTS, GET_DECISION_TREE_RESULTS
from model import GET_FFNBOW_RESULTS

#INPUT_DATA_NAME = '8K_data_short.tsv'
INPUT_DATA_NAME = '8K_data.tsv'
PICKLED_YFINANCE = 'TIKR_DATA.pkl'
HOLD_PERIOD=90

EVAL_YEAR = 2018
EVAL_YEAR = datetime.strptime(str(EVAL_YEAR) + '0101', '%Y%m%d')
model_savepath = 'test1'

K = 2
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

model = GET_DECISION_TREE_RESULTS
model_savepath = f'RandomForest_test1'
max_depth = 80
n_estimators = 10
max_features = None
splitter = "random"
min_samples_split = 50
min_df = 0.2
max_df = 0.8
stop_words = "english"

# model = GET_FFNBOW_RESULTS
all_metrics = []
output_df = pd.DataFrame()
N_YEARS = 2023-EVAL_YEAR.year+1
for idx, (traindf, validationdf, testdf) in enumerate(
            generate_data_splits(data, strategy=strategy,
                                 periods_to_test=N_YEARS)):

    # x_train, y_train = traindf['text'], traindf['label']
    # x_test, y_test = testdf['text'], testdf['label']
    
    for k, (train_index, test_index) in enumerate(KFold(n_splits=K, shuffle=True).split(traindf)):
        traindf_k = traindf.iloc[train_index]

        """
        total_test = len(y_test)
        total_train = len(y_train)
        for idx, label in enumerate(['negative', 'positive', 'neutral']):
            prop_train = sum(y_train == idx)/total_train
            prop_test = sum(y_test == idx)/total_test
            print(f"{label} - test: {prop_test:0.4f}, train: {prop_train:0.4f}")
        """

        out, metrics = model(
            traindf_k, validationdf, testdf,
            max_depth=max_depth, threshold=0.9,
            out_inplace=False, is_decision_tree = False,
            n_estimators = n_estimators, max_features= max_features, 
            max_leaf_nodes =None, min_samples_split = min_samples_split,
            splitter = splitter, min_df = min_df,
            max_df = max_df, stop_words = stop_words)
        out['Date'] = pd.to_datetime(out['Date'], format="%Y%m%d")
        metrics.year = testdf.Date.iloc[0].year
        metrics.k = k + 1
        print(f'{metrics.year}, k = {k+1}')
        print(metrics)
        all_metrics.append(metrics)

        # get_reference_data(out, yd, cols=['Annual Return', 'beta', 'sp Annual', 'sp Percent'])

        temp_df = pd.DataFrame()
        temp_df[['tikr', 'Date','label', 'pred', 'score']] = out[['tikr', 'Date','label', 'pred', 'score']]

        temp_df['pred'] = temp_df['pred'].astype(int)
        temp_df['year'] = [metrics.year]*len(out)
        temp_df['k'] = [k+1]*len(out)
        output_df = pd.concat([output_df, temp_df])

if not os.path.exists('model_outputs'):
    os.makedirs('model_outputs')
output_df.reset_index(drop=True).to_csv(f'model_outputs/outputs_{model_savepath}.csv')