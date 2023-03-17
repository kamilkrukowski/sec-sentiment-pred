import os
from datetime import datetime
import pickle
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import yfinance as yf


def get_tikrs(file_dir):
    '''
    Reads in the TIKRS from a txt file separted by newlines
    '''
    with open(file_dir) as f:
        tikrs = f.read().splitlines()
    return tikrs


def load_data(tikrs, moving_period):
    '''
    Loads in historical data by tikr into a dictionary
    tikrs and calculates specified avg moving period
    '''
    TIKR_dict = {}
    for tikr in tqdm(tikrs, desc="Downloading historical TIKR price data"):
        company_df = load_df(tikr, moving_period)
        TIKR_dict[tikr] = company_df
    TIKR_dict['^GSPC'] = load_df('^GSPC', moving_period)
    return TIKR_dict


def load_df(tikr, n_moving_avg):
    '''
    Downloads and clean dataframe and calculates specified avg moving period
    '''
    df = yf.download(tikr, start="1990-01-01", progress=False).reset_index()
    df['Close'] = df['Close'].replace('[\$,]', '', regex=True).astype(float)
    df['n_moving_avg'] = df['Close'].rolling(window=n_moving_avg).mean()

    return df


def label_performance(tikr, ref_df, date, days_period, threshold=0):
    '''
    Assigns Label to TIKR and Date

    Parameters
        ---------
        tikr: str
            a company identifier
        ref_df: pd.df
            reference for market performance (S&P 500)
        date: int
            date of document to label
        days_period: int
            number of trading days after date to compare performance
        threshold: int
            percent more or less to label underperformance or overperformance
    '''
    try:
        date = datetime.strptime(str(date), "%Y%m%d")
    except:
        return -1  # date formatting isue
    # gets n-day moving average from day before
    try:
        company_df = TIKRS_dat[tikr]
        company_avg_before = company_df[company_df['Date']
                                        < date].iloc[-1]['n_moving_avg']
        # gets n-day moving average from days_period after date
        company_avg_after = company_df[company_df['Date']
                                       > date].iloc[days_period]['n_moving_avg']

        ref_avg_before = ref_df[ref_df['Date'] < date].iloc[-1]['n_moving_avg']

        ref_avg_after = ref_df[ref_df['Date'] >
                               date].iloc[days_period]['n_moving_avg']
        company_delta = (company_avg_after -
                         company_avg_before) / company_avg_before
        ref_delta = (ref_avg_after - ref_avg_before) / ref_avg_before

        if company_delta < ref_delta - threshold:
            return 0  # underperforms

        elif company_delta > ref_delta + threshold:
            return 1  # outperforms
        else:
            return 2  # neutral performance
    except:
        return -1  # invalid date or error

def get_price(tikr, date, days_after):
    try:
        date = datetime.strptime(str(date), "%Y%m%d")
        company_df = TIKRS_dat[tikr]
        price_n_days_after = company_df[company_df["Date"]
                                    > date].iloc[days_after-1]['n_moving_avg']
        return price_n_days_after
    except:
        return -1 # date too old/new or wrong format
    

   

def load_historical_data(filename):
    with open(filename, 'rb') as handle:
        tikr_dict = pickle.load(handle)
        return tikr_dict


def output_historical_data(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


'''
DOWNLOADS TIKR DATA TO STORE
Only run if TIKR_DATA.pickle does not exist
'''

tikrs = get_tikrs('tikrs.txt')
if not os.path.exists('TIKR_DATA.pickle'):
    TIKRS_dat = load_data(tikrs, 7)
    output_historical_data('TIKR_DATA.pickle', TIKRS_dat)

else:
    TIKRS_dat = load_historical_data('TIKR_DATA.pickle')
    for tikr in tikrs:
        if tikr not in TIKRS_dat:
            print(f"Loading: {tikr}")
            TIKRS_dat[tikr] = load_df(tikr, 7)

        



#####################################################

# loads historical dataframe from pickle file
# TIKRS_dat = load_historical_data('TIKR_DATA.pickle')


data = pd.read_csv('8k_data.tsv', sep='\t')
data['label'] = data.apply(lambda row: label_performance(
    row['tikr'], TIKRS_dat['^GSPC'], row['Date'], 7, 0.01), axis=1)

data['1_day_after_moving_avg'] = data.apply(lambda row: get_price(
    row['tikr'], row['Date'], 1), axis=1)

data['7_day_after_moving_avg'] = data.apply(lambda row: get_price(
    row['tikr'], row['Date'], 7), axis=1)

data['30_day_after_moving_avg'] = data.apply(lambda row: get_price(
    row['tikr'], row['Date'], 30), axis=1)

data['90_day_after_moving_avg'] = data.apply(lambda row: get_price(
    row['tikr'], row['Date'], 90), axis=1)


print(data['label'].value_counts())

print(data.head(10))
data.to_csv('8k_data_labels.tsv', sep='\t')

