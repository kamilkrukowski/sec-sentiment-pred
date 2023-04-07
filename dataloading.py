from collections import UserDict
import os
import pickle
import warnings
from datetime import datetime, timedelta
from typing import Iterable
from argparse import ArgumentParser

import pandas as pd
import numpy as np
import yfinance as yf
from tqdm.auto import tqdm

from utils import prog_read_csv


class HistoricalYahoo(UserDict):

    def __init__(self, save_path='TIKR_DATA.pkl',
                 usecols=['Adj Close', 'Date', 'Volume'],
                 min_period_days=210):
        super().__init__()
        self.save_path = save_path
        self.historical_dir = 'historical'
        self.usecols = usecols

        self.dropped = set()

        self.min_period_days = min_period_days

    def __getitem__(self, key):
        if not isinstance(key, str):
            raise TypeError("Must index with string Company Stock Ticker")
        key = key.upper()
        if key not in self:
            res_ = self.get_company_df(key)
            if res_.empty:
                return None
        return super().__getitem__(key)

    def preload_tikrs(self, data, bar=True, desc='Downloading Tikrs'):
        """
        Preload tikrs onto self from dataframe.
        Will delete all tikrs from dataframe that are too short for self.
        """
        tikrs = np.unique(list(data.tikr) + ['^GSPC'])
        updated_ = False
        dropped = set()
        for tikr in tikrs:
            if tikr in self:
                continue  # Already loaded, no need to reload
            elif tikr in self.dropped:
                # We don't load these, since they've been dropped already
                data.drop(data[data.tikr == tikr].index, inplace=True)
                dropped.add(tikr)
                continue
            else:
                # We try to load these, maybe add them to dropped list
                out_ = self.get_company_df(tikr, return_short=False)
                if out_ is None:
                    data.drop(data[data.tikr == tikr].index, inplace=True)
                    dropped.add(tikr)
                    self.dropped.add(tikr)

                updated_ = True

        if len(dropped) != 0:
            print(f"Dropped [ {', '.join(dropped)} ]")

        if updated_:
            self.save_all()
            self.cache()

    def get_company_df(self, tikr, return_short=True):
        '''
        Downloads and clean dataframe.
        Load from cache if already downloaded.

        Parameters
        ----------
        return_short: bool
        If false, then 'new companies' will be replaced by None in return
        '''
        assert isinstance(tikr, str)

        df = None

        tikr_ = tikr.replace('^', 'X_').upper()
        save_dir = os.path.join(self.historical_dir, tikr_ + '.csv')
        if not os.path.exists(save_dir):
            df = yf.download(tikr, start="1990-01-01",
                             progress=False).reset_index()
            df['Close'] = df['Close'].replace(
                '[\\$,]', '', regex=True).astype(float)
            df.to_csv(save_dir)
        else:
            df = pd.read_csv(save_dir, usecols=self.usecols)

        if 'Date' in self.usecols:
            df['Date'] = pd.to_datetime(df['Date'], format="%Y-%m-%d")

        if not df.empty:
            if len(df) > self.min_period_days:
                self[tikr] = df
            elif len(df) < self.min_period_days and not return_short:
                return None

        return df

    def cache(self):
        """Cache class object state with onloaded data."""
        with open(self.save_path, 'wb') as handle:
            pickle.dump(self, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def save(self, tikr: str):
        """Save tikr data to csv file."""
        tikr_ = tikr.replace('^', 'X_')
        save_dir = os.path.join(self.historical_dir, tikr_ + '.csv')
        self[tikr].to_csv(save_dir)

    def save_all(self):
        """Save all held company data to separate csv files."""
        for tikr in self:
            self.save(tikr)

    @staticmethod
    def load(path):
        """Load class instance from cache file."""
        if not os.path.exists(path):
            warnings.warn(f"No Historical Data at {path}")
            return HistoricalYahoo(save_path=path)
        with open(path, 'rb') as handle:
            return pickle.load(handle)

    def map(self, func, keys=None):
        """Like pandas.apply but to all child dataframes."""
        return self.progress_map(func=func, keys=keys, bar=False)

    def progress_map(self, func, keys=None, on='df', bar=True,
                     desc='Applying...', leave=True):
        """
        Apply function to each dictionary element, and set element to result.

        Parameters
        ----------
        on: {'df', 'tikrs'}
        The object to apply the operation on. Either DataFrame or String.
        """
        itera = None
        if keys is None:
            itera = self.keys()
        else:
            if not isinstance(keys, Iterable):
                raise TypeError(
                    f"Keys are not iterable type: {str(type(keys))}")
            if len(keys) == 0:
                return
            itera = keys
        if bar:
            itera = tqdm(itera, desc=desc, leave=leave)
        for i in itera:
            inp = i
            if on == 'df':
                inp = self[i]
            self[i] = func(inp)

    def filter(self, func):
        return [i for i in self if func(self[i])]

    def __str__(self):
        tikrs = sorted(list(self.keys()))
        nkeys = len(tikrs)
        if nkeys < 10:
            TIKR_SUMMARY = ','.join(tikrs)
        else:
            TIKR_SUMMARY = ', '.join(
                tikrs[:3]) + '  ...  ' + ', '.join(tikrs[-3:])
        return f'Historical Yahoo\nLoaded {nkeys} Companies: [{TIKR_SUMMARY}]'

    def __repr__(self):
        return str(self)


def annualize_column(df, period: float, inname='Percent Return',
                     outname='Annual Return', addone=True):
    """
    Add DataFrame Column with annualized percent return.

    Parameters
    ----------
    return_period: float
        The number of days used in calculating Percent Return
    """
    exp_ = 365.25 / period
    addone = int(addone)
    df[outname] = np.power(df[inname] + addone, exp_) - addone
    return df


def add_return(df, period=1, outname='Percent Return',
               step=1, inname='Adj Close', dropna=False):
    """Calculate percent returns over period

    Parameters
    ----------
    period: int
        The number of trading days over which return is calculated
    step: int
        The stride size between samples. step=1 returns all possible windows.
        step=2 only returns every other possible return period. For
        non-overlapping returns, step must be equal to period."""
    data = df[[inname]]
    next_ = data.copy()
    next_.index -= period
    return_ = ((next_ - data) / data).iloc[period:-period]
    df[outname] = return_

    if dropna:
        df.drop(df.tail(period).index, inplace=True)

    if step != 1:
        all_ = set(df.index)
        keep_ = set(range(df.index.min(), df.index.max() + 1, step))
        remove_ = all_ - keep_
        df.drop(remove_, inplace=True)

    return df


def label_sentiment(df, ref, threshold=0.00, period=1, outname='Outlook',
                    inname='Annual Return'):

    POS_CLASS = 1
    NEG_CLASS = 0
    NEUTRAL_CLASS = 2

    df['index'] = df.index
    df.set_index('Date', inplace=True)

    if threshold == 0.0:
        df[outname] = df[inname].ge(ref).astype(int)
    else:
        pos_plus = df.ge(ref + threshold)
        pos_minus = df.ge(ref - threshold)

        outlook = pos_plus.astype('int')
        # If a sentiment is neutral, it changed here and difference is 1.
        # We use 2 as index for neutral sentiment
        outlook += NEUTRAL_CLASS * (pos_plus - pos_minus)

    df['Date'] = df.index
    df.set_index('index', inplace=True)
    return df


def label_filingdates(df, dates):
    # Ensure we never get 'date not found' error from indexing.
    dates = pd.merge(dates, df.Date).Date

    df['index'] = df.index
    df.set_index('Date', inplace=True)

    df['HAS_FILING'] = False
    df.loc[np.array(dates), 'HAS_FILING'] = True

    df['Date'] = df.index
    df.set_index('index', inplace=True)

    return df


def get_betas(x_name, window, returns_data):
    """
    Parameters
    ----------
    x_name: str
        The column name to use as market
    window: int
        The number of elements to consider for beta
    returns_data: pandas.DataFrame
        The dataframe of percent returns to consider
    """
    window_inv = 1.0 / window
    x_sum = returns_data[x_name].rolling(window, min_periods=window).sum()
    y_sum = returns_data.rolling(window, min_periods=window).sum()
    xy_sum = returns_data.mul(
        returns_data[x_name], axis=0).rolling(
        window, min_periods=window).sum()
    xx_sum = np.square(
        returns_data[x_name]).rolling(
        window,
        min_periods=window).sum()
    xy_cov = xy_sum - window_inv * y_sum.mul(x_sum, axis=0)
    x_var = xx_sum - window_inv * np.square(x_sum)
    betas = xy_cov.divide(x_var, axis=0)[window - 1:]
    betas.columns.name = None
    return betas


def add_beta(df, ref, window, hold_period=1, start_date=None,
             silent=False) -> float:

    inname = f'Percent Return ({hold_period})'

    df1 = df[[inname, 'Date']].copy().rename(
        columns={inname: 'DF1'})

    ref = ref[[inname, 'Date', 'Annual Return', 'Percent Return']]

    if start_date is not None:
        df = df[df.Date > start_date]

    # Find intersection of dates, then re-split this subset in add_return.
    merged_ = pd.merge(
        df1, ref, how='inner', on=['Date'])
    keys = ['Date', 'Annual Return', 'Percent Return']
    dropped = merged_[keys][window-1:]
    merged_ = merged_.drop(keys, axis=1)

    betas = None
    if window is None:
        betas = get_betas(inname, len(merged_), merged_)
    elif len(merged_) < window + 1:
        if not silent:
            warnings.warn(
                f"window SIZE {window} is greater than "
                f"available DATA length {len(merged_)}",
                RuntimeWarning)
        betas = get_betas(inname, len(merged_), merged_)
    else:
        betas = get_betas(inname, window, merged_)

    indices = df.set_index(
        'Date').index.get_indexer(
        list(dropped['Date'].values))

    df['beta'] = -999
    beta_idx = df.columns.get_loc("beta")
    df.iloc[indices, beta_idx] = betas['DF1'].values

    df['sp Annual'] = -999
    sp_ann_idx = df.columns.get_loc('sp Annual')
    df.iloc[indices, sp_ann_idx] = dropped['Annual Return'].values

    df['sp Percent'] = -999
    sp_perc_idx = df.columns.get_loc('sp Percent')
    df.iloc[indices, sp_perc_idx] = dropped['Percent Return'].values

    return df


def add_betas(yd, window, hold_period=1, start_date=None, silent=False):
    """
    Parameters
    ----------
    window: int
        The historical length in days for beta calculation. Typical 1Y or 5Y
    """
    # Add missing Percent Return
    filtered = yd.filter(
        lambda x: f'beta' not in x.columns)
    ref = yd['^GSPC']
    yd.progress_map(lambda x: add_beta(x, ref, window=window,
                                       hold_period=hold_period, start_date=start_date, silent=True),
                    keys=filtered, desc='Generating Beta', bar=not silent)


def add_percent_return(yd, hold_period, dropna=False, silent=False,
                       desc='Generating Percent Return'):
    # Add missing Percent Return
    filtered = yd.filter(
        lambda x: f'Percent Return ({hold_period})' not in x.columns)
    yd.progress_map(lambda x: add_return(
        x, period=hold_period, outname=f'Percent Return ({hold_period})',
        dropna=dropna), bar=not silent,
        keys=filtered, desc=desc)


def annualize_metric(yd, hold_period, addone=True, silent=False):
    # Add missing Annual Return
    filtered = yd.filter(
        lambda x: f'Annual Return ({hold_period})' not in x.columns)
    yd.progress_map(lambda x: annualize_column(
        x, period=hold_period, outname=f'Annual Return ({hold_period})',
        addone=addone, inname=f'Percent Return ({hold_period})'),
        keys=filtered, desc='Annualizing Returns', bar=not silent)


def add_outlook(yd, hold_period, silent=False):
    # Add missing sentiment class
    filtered = yd.filter(lambda x: 'Outlook' not in x.columns)
    filtered = set(filtered) - {'^GSPC'}
    yd.progress_map(lambda x: label_sentiment(
                    x, yd['^GSPC'].set_index(
                        'Date')[f'Annual Return ({hold_period})'],
                    period=hold_period, inname=f'Annual Return ({hold_period})'),
                    keys=filtered, desc="Classifying Sentiment of Returns",
                    bar=not silent)


def add_has_filing(data, yd, silent=False):
    # Label which element has a potential 8-K filing
    filtered = yd.filter(lambda x: 'HAS_FILING' not in x.columns)
    yd.progress_map(lambda x: label_filingdates(
                    yd[x], data[data.tikr == x].Date),
                    keys=filtered, on='tikrs', desc='Annotating Filings',
                    bar=not silent)


def add_sp(yd, silent=False, leave=False):
    # Label which element has a potential 8-K filing
    filtered = yd.filter(lambda x: 'sp Annual Return' not in x.columns)
    yd.progress_map(lambda x: label_filingdates(
                    yd[x], data[data.tikr == x].Date),
                    keys=filtered, on='tikrs', desc='Annotating Filings',
                    bar=not silent, leave=leave)


def get_reference_data(data, yd, cols=['Outlook'],
                       desc='Looking Up Labels for Text',
                       backfill=True, leave=False):
    """
    Takes cols from yd historical dataframe and add them to data
    Will join based on indexing by 'Date' on data.
    """
    if isinstance(cols, str):
        cols = [cols]
    assert isinstance(cols, list)

    assert 'tikr' in data.columns
    assert 'Date' in data.columns

    for tikr in yd:
        df = yd[tikr]
        df['index'] = df.index
        df['__DATE__'] = df['Date']
        df.set_index('__DATE__', inplace=True)

    def applyfunc_backfill(row, innames):
        df = yd[row['tikr']]
        # print(row['tikr'])
        # Backfill finds nearest trading date after news release
        indexer = df.index.get_indexer([row['Date']], method='backfill')
        entry = df.iloc[indexer]
        return entry[innames].values[0, :]

    def applyfunc_no_backfill(row, innames):
        # print(row['tikr'])
        df = yd[row['tikr']]
        # Backfill finds nearest trading date after news release
        indexer = df.index.get_indexer([row['Date']])
        entry = df.iloc[indexer]
        return entry[innames].values[0, :]

    applyfunc = applyfunc_no_backfill
    if backfill:
        applyfunc = applyfunc_backfill

    tqdm.pandas(desc=desc, leave=leave)
    arr = data.progress_apply(lambda x: applyfunc(x, innames=cols),
                              axis=1, result_type='expand')
    data[cols] = pd.DataFrame(arr)

    for tikr in yd:
        df = yd[tikr]
        df.set_index('index', inplace=True)


def calculate_metrics(yd, hold_period, dropna=False, silent=False):

    add_percent_return(yd, hold_period=hold_period,
                       dropna=dropna, silent=silent)
    annualize_metric(yd, hold_period=hold_period, silent=silent)
    add_outlook(yd, hold_period=hold_period, silent=silent)

    keyann = f'Annual Return ({hold_period})'
    keyperc = f'Percent Return ({hold_period})'

    def set_without_period(df, keyann, keyperc):
        df['Annual Return'] = df[keyann]
        df['Percent Return'] = df[keyperc]
        return df

    yd.map(lambda x: set_without_period(x, keyann, keyperc))

    # Daily Beta Calculation
    add_percent_return(yd, hold_period=1, silent=silent,
                       desc='Generating Daily Returns')
    add_betas(yd, 365, silent=silent)


HOLD_PERIOD = 90
RAW_DATA_NAME = '8k_data_filtered'
OUTPUT_NAME = '8k_data'
PICKLED_YFINANCE = 'TIKR_DATA.pkl'
START_DATE = datetime.strptime('2000-01-01', '%Y-%m-%d')
END_DATE = datetime.strptime(
    '2023-01-01', '%Y-%m-%d') - timedelta(days=HOLD_PERIOD)
N_MOVING_AVG = 7

yd = None
data = None
if __name__ == '__main__':
    args = ArgumentParser()
    args.add_argument(
        '-nsamples',
        '-n',
        action='store',
        default=None,
        type=int)
    args = args.parse_args()

    keys = ['Date', 'tikr', 'text']
    data = prog_read_csv(f'{RAW_DATA_NAME}.tsv', sep='\t', usecols=keys,
                         nrows=args.nsamples, desc='Loading 8K Data')
    data['Date'] = pd.to_datetime(data['Date'], format="%Y-%m-%d")

    yd = HistoricalYahoo.load(PICKLED_YFINANCE)
    yd.preload_tikrs(data)

    calculate_metrics(yd, hold_period=HOLD_PERIOD, dropna=True)

    add_has_filing(data, yd)  # Identify subset with filings

    """
    Sometimes the document release date is NOT a trading date.
    This makes cross-referencing hard, as we have to check the first
    trading date AFTER the release date. Here we annotated with the earliest
    trading day relative to the release date (Hopefully same day), which speeds
    up future calculations.
    """
    data['Release Date'] = data['Date']
    get_reference_data(data, yd, cols=[
        'Date', 'Outlook', 'Percent Return', 'Annual Return', 'sp Annual', 'sp Percent'],
        backfill=False)
    data.rename(columns={'Outlook': 'label'}, inplace=True)
    data['Trading Date'] = data['Date']
    data.label.value_counts()

    yd.cache()
    print("Sort by Date and Offload Results")
    data.sort_values(
        by='Date', ascending=False).to_csv(OUTPUT_NAME + '.tsv', sep='\t')
