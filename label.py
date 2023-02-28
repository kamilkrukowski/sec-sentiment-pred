import numpy as np
import pandas as pd
from datetime import datetime, timedelta

tikrs = ['aapl', 'msft', 'amzn', 'tsla', 'googl', 'goog',  'unh', 'jnj', 'cvx',
         'pg', 'jpm', 'hd', 'v']

def load_data(tikrs, moving_period):
    TIKR_dict = {}
    for tikr in tikrs:
        data_dir = str('historical/' + tikr + str('.csv'))
        company_df = clean_df(data_dir, moving_period)
        TIKR_dict[tikr] = company_df
    return TIKR_dict


def clean_mine(file_dir, moving_avg_d):
    df = pd.read_csv(file_dir)
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(str(x).replace('/', ''), '%Y%m%d'))
    df['Close/Last'] = df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
    df['moving_avg'] = df['Close/Last'].rolling(window=moving_avg_d).mean()

def clean_df(file_dir, moving_avg_d):
    df = pd.read_csv(file_dir)
    df['Date'] = df['Date'].apply(lambda x: datetime.strptime(str(x).replace('/', ''), '%m%d%Y'))
    df['Close/Last'] = df['Close/Last'].replace('[\$,]', '', regex=True).astype(float)
    df['moving_avg'] = df['Close/Last'].rolling(window=moving_avg_d).mean()
    return df

def label_performance(tikr, ref_df, date, days_period, threshold=0):
    try:
        date = datetime.strptime(str(date).replace('/',''), "%Y%m%d")
    except:
        return -3 #date formatting isue
    # gets n-day moving average from day before
    try:
        company_df = TIKRS_dat[tikr]
        company_avg_before = company_df[company_df['Date'] < date].iloc[0]['moving_avg']
        # gets n-day moving average from days_period after date
        company_avg_after = company_df[company_df['Date'] > date].iloc[-days_period]['moving_avg']
        ref_avg_before = ref_df[ref_df['Date'] < date].iloc[0]['moving_avg']
        ref_avg_after = ref_df[ref_df['Date'] > date].iloc[-days_period]['moving_avg']

        company_delta = (company_avg_after - company_avg_before) / company_avg_before
        ref_delta = (ref_avg_after - ref_avg_before) / ref_avg_before

        if company_delta < ref_delta - threshold:
            return 0 # underperforms

        elif company_delta > ref_delta + threshold:
            return 1 # outperforms
        else:
            return 2 # neutral performance
    except:
        return -1 # this means it was unsuccessful and there was an error processing


tikrs = tikrs[:1]

TIKRS_dat = load_data(tikrs, 7)
comp_his = clean_df('historical/comp.csv', 7)

data = pd.read_csv('8k_data.csv')
data['label'] = data.apply(lambda row: label_performance(
            row['TIKR'], comp_his, row['Date'], 7, 0.01), axis=1)

pd.save_csv('8k_data_labels.csv')