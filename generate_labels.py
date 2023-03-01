import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf

def get_tikrs(file_dir):
    with open(file_dir) as f:
        tikrs = f.read().splitlines()
    return tikrs

def load_data(tikrs, moving_period):
    TIKR_dict = {}
    for tikr in tikrs:
        data_dir = str('historical/' + tikr + str('.csv'))
        company_df = clean_df(data_dir, moving_period)
        TIKR_dict[tikr] = company_df
    return TIKR_dict


def clean_df(tikr, moving_avg_d):
    df = yf.download(tikr, start="1990-01-01").reset_index()
    df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)
    df['moving_avg'] = df['Close'].rolling(window=moving_avg_d).mean()
    return df

def label_performance(tikr, ref_df, date, days_period, threshold=0):
    try:
        date = datetime.strptime(str(date), "%Y%m%d.%f")
    except:
        return -1 #date formatting isue
    # gets n-day moving average from day before
    try:
        company_df = TIKRS_dat[tikr]
        company_avg_before = company_df[company_df['Date'] < date].iloc[-1]['moving_avg']
        # gets n-day moving average from days_period after date
        company_avg_after = company_df[company_df['Date'] > date].iloc[days_period]['moving_avg']

        ref_avg_before = ref_df[ref_df['Date'] < date].iloc[-1]['moving_avg']

        ref_avg_after = ref_df[ref_df['Date'] > date].iloc[days_period]['moving_avg']
        company_delta = (company_avg_after - company_avg_before) / company_avg_before
        ref_delta = (ref_avg_after - ref_avg_before) / ref_avg_before

        if company_delta < ref_delta - threshold:
            return 0 # underperforms

        elif company_delta > ref_delta + threshold:
            return 1 # outperforms
        else:
            return 2 # neutral performance
    except:
        return -1


tikrs = get_tikrs('100_tikrs.txt')
TIKRS_dat = load_data(tikrs, 7)
comp_his = clean_df('^GSPC', 7)


data = pd.read_csv('8k_data.csv')
data['label'] = data.apply(lambda row: label_performance(
            row['tikr'], comp_his, row['Date'], 7, 0.01), axis=1)

data.to_csv('8k_data_labels.csv')


# exporting to pickle https://stackoverflow.com/questions/11218477/how-can-i-use-pickle-to-save-a-dict-or-any-other-python-object