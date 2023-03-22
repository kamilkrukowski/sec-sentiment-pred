import os
from datetime import datetime
import pickle
from argparse import ArgumentParser


import numpy as np
import pandas as pd
import yfinance as yf


from utils import prog_read_csv

HOLD_PERIOD = 90
TIKR_LIST_FILE = 'tikr.txt'
PICKLED_YFINANCE = 'TIKR_DATA.pickle'

args = ArgumentParser()
args.add_argument('--demo', action='store_true')
args = args.parse_args()


def load_historical_data(tikrs, moving_period):
    '''
    Loads in historical data by tikr into a dictionary
    tikrs and calculates specified avg moving period
    '''
    TIKR_dict = {}
    for tikr in tqdm(tikrs, desc="Downloading historical TIKR price data"):
        company_df = get_company_df(tikr, moving_period)
        TIKR_dict[tikr] = company_df
    TIKR_dict['^GSPC'] = get_company_df('^GSPC', moving_period)
    return TIKR_dict


def get_company_df(tikr, n_moving_avg):
    '''
    Downloads and clean dataframe and calculates specified avg moving period
    '''
    df = yf.download(tikr, start="1990-01-01", progress=False).reset_index()
    df['Close'] = df['Close'].replace('[\\$,]', '', regex=True).astype(float)
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
    if isinstance(date, int):
        date = datetime.strptime(str(date), "%Y%m%d")
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
    except BaseException:
        return None


def n_days_annualized_return(
        tikr, start_date, days_period, inflation_adjusted=False):
    '''
    Calculates the n-day annualized return for a given TIKR and start date

    Parameters
        ---------
        tikr: str
            a company identifier
        start_date: int or datetime object
            start date of the period to calculate return
        days_period: int
            number of trading days to include in the period
        inflation_adjusted: bool
            flag to indicate whether to adjust returns for inflation

    Returns
        -------
        annualized_return: float
            the n-day annualized return as a percentage
    '''
    if isinstance(start_date, int):
        start_date = datetime.strptime(str(start_date), "%Y%m%d")
    try:

        start_date = start_date.strftime('%Y-%m-%d')

        company_df = TIKRS_dat[tikr]
        start_price = company_df[company_df['Date']
                                 > start_date].iloc[0]['Close']
        end_price = company_df[company_df['Date'] >
                               start_date].iloc[days_period - 1]['Close']

        # Calculate the fractional number of years using the total number of
        # days
        n = days_period / 365.25  # assuming a leap year every 4 years
        if inflation_adjusted:
            inflation_rate = calculate_inflation(
                start_date, days_period, silence=True)
            return ((end_price / start_price) /
                    (1 + inflation_rate)) ** (1 / n) - 1
        else:
            # print(f"{start_date}: {start_price}, {end_date}: {end_price}")
            return (end_price / start_price) ** (1 / n) - 1
    except BaseException:
        return None


def n_days_percent_return(
        tikr, start_date, days_period, inflation_adjusted=False):
    '''
    Calculates the n-day annualized return for a given TIKR and start date

    Parameters
        ---------
        tikr: str
            a company identifier
        start_date: int or datetime object
            start date of the period to calculate return
        days_period: int
            number of trading days to include in the period
        inflation_adjusted: bool
            flag to indicate whether to adjust returns for inflation

    Returns
        -------
        annualized_return: float
            the n-day annualized return as a percentage
    '''
    if isinstance(start_date, int):
        start_date = datetime.strptime(str(start_date), "%Y%m%d")
    try:
        start_date = start_date.strftime('%Y-%m-%d')

        company_df = TIKRS_dat[tikr]
        start_price = company_df[company_df['Date']
                                 > start_date].iloc[0]['Close']
        end_price = company_df[company_df['Date'] >
                               start_date].iloc[days_period - 1]['Close']

        # Calculate the fractional number of years using the total number of
        # days
        n = days_period / 365.25  # assuming a leap year every 4 years
        if inflation_adjusted:
            inflation_rate = calculate_inflation(
                start_date, days_period, silence=True)
            return ((end_price / start_price) /
                    (1 + inflation_rate)) - 1
        else:
            return (end_price / start_price) - 1
    except BaseException:
        return None


def generate_annualized_return(tikr, start_date, n_days=[7, 30, 90],
                               inflation_adjusted=False):
    """
    Calculates the annualized return for a given company TIKR and the S&P500 index for the specified number of days
    starting from the given start date. Returns a tuple of annualized returns for the specified periods, and for both
    the company and S&P500 index.

    Parameters:
    -----------
    tikr: str
        A string representing the TIKR (ticker) symbol of the company.
    start_date: int or datetime.datetime object
        The start date of the period for which returns are to be calculated. This can be either an integer in the format
        YYYYMMDD or a datetime.datetime object.
    n_days: List[int]
        A list of integers representing the number of days for which annualized returns are to be calculated. The default
        value is [7, 30, 90].
    inflation_adjusted: bool
        A boolean flag indicating whether to adjust the returns for inflation. The default value is False.

    Returns:
    --------
    A tuple of floats representing start price of next day, the annualized returns for the specified periods
    and for both the company and S&P500 index. The tuple has a length of 3 times the length of `n_days`.
    """
    try:
        # Convert start_date to a datetime object if it's an integer
        if isinstance(start_date, int):
            start_date = datetime.strptime(str(start_date), "%Y%m%d")

        # Initialize an empty tuple to hold the calculated returns
        result = ()

        # Convert start_date to string format for use in DataFrame indexing
        start_date = start_date.strftime('%Y-%m-%d')

        # Retrieve the DataFrame for the given company and S&P500 from the
        # TIKRS_dat dictionary
        company_df = TIKRS_dat[tikr]
        sp_df = TIKRS_dat['^GSPC']

        # Get the stock prices for the given company and S&P500 on the start
        # date
        start_price = company_df[company_df['Date']
                                 > start_date].iloc[0]['Close']
        sp_start_price = sp_df[sp_df['Date'] > start_date].iloc[0]['Close']

    except BaseException:
        # If there's an error in the try block, return None for all the
        # calculated returns
        return (None,) * (len(n_days) * 4)
    # Calculate the returns for each specified period and add them to the
    # result tuple
    for n in n_days:
        try:
            # Get the stock prices for the given company and S&P500 n days
            # after the start date
            n_day_price = company_df[company_df['Date']
                                     > start_date].iloc[n - 1]['Close']
            sp_n_day_price = sp_df[sp_df['Date']
                                   > start_date].iloc[n - 1]['Close']

            # Calculate the return for the given company for the specified
            # period
            if inflation_adjusted:
                # If inflation_adjusted is True, adjust the return for
                # inflation using the
                # calculate_inflation function and the start date and period
                # length (n)
                n_day_inflation_rate = calculate_inflation(
                    start_date, n, silence=True)
                n_day_return = ((n_day_price / start_price) /
                                (1 + n_day_inflation_rate)) ** (365.25 / n) - 1
                sp_n_day_return = ((sp_n_day_price / sp_start_price) /
                                   (1 + n_day_inflation_rate)) ** (365.25 / n) - 1
            else:
                # If inflation_adjusted is False, calculate the return without
                # adjusting for inflation
                n_day_return = (n_day_price / start_price) ** (365.25 / n) - 1
                sp_n_day_return = (
                    sp_n_day_price / sp_start_price) ** (365.25 / n) - 1

            result += (start_price, n_day_price, n_day_return, sp_n_day_return)
        except BaseException:
            # If an error occurs, return None
            result += (None, None, None, None)
    return result


def get_price(tikr, date, days_after):
    try:
        if isinstance(date, int):
            date = datetime.strptime(str(date), "%Y%m%d")
        company_df = TIKRS_dat[tikr]
        price_n_days_after = company_df[
            company_df["Date"] > date].iloc[days_after - 1]['n_moving_avg']
        return price_n_days_after
    except BaseException:
        return np.nan  # date too old/new or wrong format


def load_historical_data(filename):
    with open(filename, 'rb') as handle:
        tikr_dict = pickle.load(handle)
        return tikr_dict


def output_historical_data(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)


def calculate_inflation(start_date, days_period, silence=True):
    """
    average annual inflation rate between two dates using the Consumer Price Index (CPI) for All Urban Consumers:
    All Items in U.S. City Average data.

    Parameters
    ----------
    start_date : str
        The start date in the format "YYYY-MM-DD".
    days_period: int
            number of trading days to include in the period
    silence : bool
        If True, suppress print statements.

    Returns
    -------
    float
        The inflation rate between the start and end dates.
    """
    # Convert start and end dates to pandas datetime objects
    start_date = pd.to_datetime(start_date)

    end_date = start_date + pd.DateOffset(days=days_period)
    # Find the closest CPI levels to the start and end dates

    start_cpi = CPI_df.loc[CPI_df['DATE'] <= start_date].iloc[-1]['CPIAUCSL']
    end_cpi = CPI_df.loc[CPI_df['DATE'] <= end_date].iloc[-1]['CPIAUCSL']

    # Calculate inflation rate as the percentage change in CPI
    inflation_rate = (end_cpi - start_cpi) / start_cpi

    if not silence:
        print(
            f"start_cpi: { start_cpi} end_cpi: {end_cpi}\ninflation_rate: {inflation_rate}")
    return inflation_rate


'''
DOWNLOADS TIKR DATA TO STORE
Only run if PICKLED_YFINANCE does not exist
'''
if not os.path.exists(PICKLED_YFINANCE):
    tikrs = None
    with open(TIKR_LIST_FILE) as f:
        tikrs = f.read().splitlines()
    TIKRS_dat = load_historical_data(tikrs, 7)
    output_historical_data(PICKLED_YFINANCE, TIKRS_dat)

#####################################################

# loads historical dataframe from pickle file
TIKRS_dat = load_historical_data(PICKLED_YFINANCE)
CPI_df = pd.read_csv('CPIAUCSL.csv', parse_dates=["DATE"])


nrows = 50 if args.demo else None
data = prog_read_csv('8k_data.tsv', sep='\t', nrows=nrows,
                     desc='1/4 Loading Data...')
data['Date'] = pd.to_datetime(data['Date'], format="%Y%m%d")

tqdm.pandas(desc='2/4 Generating Categories...', leave=False)
data['label'] = data.progress_apply(lambda row: label_performance(
    row['tikr'], TIKRS_dat['^GSPC'], row['Date'], HOLD_PERIOD, 0.01), axis=1)

print(data['label'].value_counts())

tqdm.pandas(desc='3/4 Generating Percent Return...', leave=False)
# Annualize the return over the investment period (7, 30, 90) day
data['percent return'] = data.progress_apply(lambda row: n_days_percent_return(
    row['tikr'],
    row['Date'],
    days_period=HOLD_PERIOD,
    inflation_adjusted=True), axis=1)

tqdm.pandas(desc='4/4 Generating Annualized Return...', leave=False)
# Annualize the return over the investment period (7, 30, 90) day
data[['start price',
      'end price',
      'annualized return',
      'sp annualized return']] = data.progress_apply(
    lambda row: generate_annualized_return(
        row['tikr'],
        row['Date'],
        n_days=[HOLD_PERIOD],
        inflation_adjusted=True), axis=1, result_type="expand")

data = data.dropna()
data = data.astype({'label': 'int8'})

print(data.head(10))

data.to_csv('8k_data_labels.tsv', sep='\t')
