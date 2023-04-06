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

K = 5
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


if not os.path.exists('figs'):
            os.makedirs('figs')
def plot_rocs(metrics):

    n = len(metrics)
    fig, axes = plt.subplots(1, n, figsize=(n*3, 3))
    fig.suptitle('ROC Curves')
    for idx, mets, in enumerate(metrics):
        x, y = mets['_test_ROC']
        auroc = mets['test_auroc']
        axes[idx].plot(x, y, label=f'test {auroc:.3f} AUC')
        x, y = mets['_train_ROC']
        auroc = mets['train_auroc']
        axes[idx].plot(x, y, label=f'train {auroc:.3f} AUC')
        
        x, y = mets['_validation_ROC']
        auroc = mets['validation_auroc']
        axes[idx].plot(x, y, label=f'validation {auroc:.3f} AUC')
        
        axes[idx].plot(x, x, 'r-.')
        if(mets.k):
            axes[idx].set_xlabel(f'{mets.year}, k = {mets.k}')
        else:
            axes[idx].set_xlabel(f'{mets.year}')
        axes[idx].legend()
    fig.tight_layout()
    plt.savefig('figs/fig1.png')
    plt.clf()

    fig, axes = plt.subplots(1, len(metrics), figsize=(n*3, 3))
    fig.suptitle('Precision Recall Curves')
    for idx, mets, in enumerate(metrics):
        x, y = mets['_test_PRC']
        axes[idx].plot(x, y, label='test')
        x, y = mets['_validation_PRC']
        axes[idx].plot(x, y, label='validation')
        x, y = mets['_train_PRC']
        axes[idx].plot(x, y, label='train')
        if(mets.k):
            axes[idx].set_xlabel(f'{mets.year}, k = {mets.k}')
        else:
            axes[idx].set_xlabel(f'{mets.year}')
    fig.tight_layout()
    plt.savefig('figs/fig2.png')
    plt.clf()

plot_rocs(all_metrics)



def organize_metrics(metrics):
    year_dict = {}
    for mets in metrics:
        annual_stats = defaultdict(list)
        if mets.year not in year_dict:
            year_dict[mets.year] = annual_stats
        
        for m in mets:
            year_dict[mets.year][m].append(mets[m])

    return year_dict

def boxplot(metrics_dict, names, save_path=None):
    n_years = len(metrics_dict)
    n_names = len(names)
    fig, axes = plt.subplots(n_names, n_years, figsize=(n_years*5, n_names*6))
    fig.tight_layout()
    fig.subplots_adjust(left=0.1, bottom=0.1, right=0.95, top=0.9, wspace=0.2, hspace=0.4)
    colors = sns.color_palette(n_colors=n_names)
    k = 0
    for i, name in enumerate(names):
        for j, year in enumerate(metrics_dict):
            if not metrics_dict[year][name]:
                raise KeyError(f'{name} does not exist in provided metrics')
            sns.boxplot(metrics_dict[year][name], ax=axes[i, j], orient='vertical', palette=[colors[i]], showmeans=True)
            k = len(metrics_dict[year][name])
            axes[i, j].set_title(year)
            if j == 0:
                axes[i, j].set_ylabel(name)
    fig.suptitle(f'K-fold Metrics for K = {k}')
    if save_path:
        plt.savefig(f'{save_path}_boxplot.png')
    else:
        plt.savefig('figs/boxplot.png')

boxplot(organize_metrics(all_metrics), 
    names = ["train_auroc", "validation_auroc", "test_auroc"], save_path="figs/short_data_auroc")

boxplot(organize_metrics(all_metrics), 
    names = ["_train_acc_pos", "_validation_acc_pos", "_test_acc_pos"], save_path="figs/short_data_accuracy")

def print_evaluation_summary(metrics_dict, names):
    year_summary = {}
    for year in metrics_dict:
        year_summary.setdefault(year, {})
        for name in names:
            year_summary[year][f'{name}_average'] = np.mean(metrics_dict[year][name])
            year_summary[year][f'{name}_min'] = np.min(metrics_dict[year][name])
            year_summary[year][f'{name}_max'] = np.max(metrics_dict[year][name])
    
    for year in year_summary:
        print(year)
        for val in year_summary[year]:
            print(f'{val}: {np.round(year_summary[year][val], 3)}')
        print('\n')
    return year_summary

print_evaluation_summary(organize_metrics(all_metrics),
    names=["_train_acc_pos", "_validation_acc_pos", "_test_acc_pos"])