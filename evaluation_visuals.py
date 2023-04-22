import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt
import seaborn as sns 

from dataloading import HistoricalYahoo, calculate_metrics, get_reference_data
from metrics import Metrics
from collections import defaultdict

def build_metrics(output):
    metrics_dict = defaultdict(dict)
    all_metrics = []
    # Group the DataFrame by year and k category
    grouped = output.groupby(['year', 'k'])
    for name, group in grouped:
        metrics = Metrics()
        predictions = np.array(group['score'])
        empty = np.empty((predictions.shape[0],))
        preds = np.column_stack((empty, predictions))
        metrics.calculate(group['label'], preds, split='test')
        metrics.year = name[0]
        metrics.k = name[1]
        metrics_dict[name[0]][name[1]]= metrics
    return metrics_dict



def boxplot(data, metric_name, save_name=None):
    auroc_values = {}
    for year, values in data.items():
        auroc_values[year] = [v[metric_name] for v in values.values()]
    
    # Plot the boxplots
    fig, ax = plt.subplots()
    ax.boxplot(auroc_values.values(), showmeans=True)
    ax.set_xticklabels(auroc_values.keys())
    ax.set_xlabel('Year')
    ax.set_ylabel('AUROC')
    # plt.show()
    if save_name:
        plt.savefig(f'figs/{save_name}_boxplot.png')
    else:
        plt.savefig('figs/boxplot.png')


def plot_roc(data, save_name = None):
    num_years = len(data.keys())

    fig, axs = plt.subplots(1, num_years, figsize=(num_years * 6, 6))
    
    for i, year in enumerate(data):
        for k in data[year]:
            fpr, tpr = data[year][k]['_test_ROC']
            auroc = data[year][k]['test_auroc']
            axs[i].plot(fpr, tpr, label=f'k = {k}, AUC = {auroc:.3f}')

        axs[i].set_xlabel('False Positive Rate')
        axs[i].set_ylabel('True Positive Rate')
        axs[i].plot([0, 1], [0, 1], 'r-.')
        axs[i].set_title(f'ROC Curves (Year = {year})')
        axs[i].legend()
        
    plt.tight_layout()
    if save_name:
        plt.savefig(f'figs/{save_name}_ROC.png')
    else:
        plt.savefig('figs/ROC.png')

save_name = "RandomForest_50"
df = pd.read_csv('model_outputs/outputs_RandomForest_50.csv')
data = build_metrics(df)

boxplot(data, "test_auroc", save_name=save_name)

plot_roc(data, save_name =save_name)