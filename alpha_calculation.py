import dataloading as dl
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
def sp_percent(data,yd,hold_period,desc='Looking Up for S&P', backfill=True, leave=False):
	
    df = yd['^GSPC']
    df['index'] = df.index
    df['__DATE__'] = df['Date']
    df.set_index('__DATE__', inplace=True)

    cols = [f'Annual Return ({hold_period})', f'Percent Return ({hold_period})']
    output_cols = ['sp Annual', 'sp Percent']

    def applyfunc_backfill(row, innames):
        df = yd['^GSPC']
        # print(row['tikr'])
        # Backfill finds nearest trading date after news release
        indexer = df.index.get_indexer([row['Date']], method='backfill')
        entry = df.iloc[indexer]
        return entry[innames].values[0, :]

    def applyfunc_no_backfill(row, innames):
        # print(row['tikr'])
        df = yd['^GSPC']
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
    data[output_cols] = pd.DataFrame(arr)

    
    df = yd["^GSPC"]
    df.set_index('index', inplace=True)

INPUT_FILE = 'outputs_19.csv'
PICKLED_YFINANCE = 'TIKR_DATA.pkl'
OUTPUT_FILE = 'outputs_19_with_alpha90_3.csv'
HOLD_PERIOD = 7

data = pd.read_csv(INPUT_FILE, index_col = 0, parse_dates = ["Date"])
split_dfs = np.array_split(data, len(data) // 973180 )
data = split_dfs[17]
# filter out prediction  == 0
data = data[data.pred== 1]
yd = dl.HistoricalYahoo.load(PICKLED_YFINANCE)
yd.preload_tikrs(data)

dl.calculate_metrics(yd, hold_period=HOLD_PERIOD, dropna=True)

dl.add_has_filing(data, yd)  # Identify subset with filings
#data = data[data.tikr == "ABBV"]

data['Release Date'] = data['Date']

# adding annual return information from TIKR_DATA.pkl
dl.get_reference_data(data, yd, cols=[
        'Date', 'Outlook', 'Percent Return', 'Annual Return', 'beta'],
        backfill=False)

sp_percent(data,yd,hold_period= HOLD_PERIOD)
# Loading the risk free rate information
rf_info = pd.read_csv("Risk_free_rate.csv", parse_dates = ["Date"], index_col = "Date")

# calculate the product once and pass it as an argument to the function
rf_cumprod = (1+rf_info/100).cumprod() 
rf_date, cumprod_rf  =  rf_info.index, rf_cumprod.values.tolist()


# based on risk free rate calculate 90 days alpha values
data[f"jensen alpha ({HOLD_PERIOD})"] = data.progress_apply(lambda x: dl.calculate_jensen_alpha(x, rf_date, cumprod_rf, inname = "Percent Return",  hold_period = HOLD_PERIOD), axis=1, result_type='expand')
data[f"simple alpha ({HOLD_PERIOD})"] = data.progress_apply(lambda x: dl.calculate_simple_alpha(x, rf_date, cumprod_rf, inname = "Percent Return",  hold_period = HOLD_PERIOD), axis=1, result_type='expand')

yd.cache()
data.to_csv(OUTPUT_FILE)