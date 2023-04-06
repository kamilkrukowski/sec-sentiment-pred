from datetime import datetime, timedelta
from argparse import ArgumentParser

from tqdm.auto import tqdm
import pandas as pd

from utils import prog_read_csv

HOLD_PERIOD = 90
RAW_DATA_NAME = '8k_data_raw'
OUTPUT_NAME = '8k_data_filtered'
PICKLED_YFINANCE = 'TIKR_DATA.pkl'
START_DATE = datetime.strptime('2010-01-01', '%Y-%m-%d')
END_DATE = datetime.strptime('2022-12-31', '%Y-%m-%d')
N_MOVING_AVG = 7
MAX_WORD_COUNT = 8000

args = ArgumentParser()
args.add_argument('-nsamples', '-n', action='store', default=None, type=int)
args = args.parse_args()


data = prog_read_csv(f'{RAW_DATA_NAME}.tsv', sep='\t',
                        nrows=args.nsamples, desc='Loading Data...')
data['Date'] = pd.to_datetime(data['Date'], format="%Y%m%d")

original_len = len(data)
data = data[data.Date >= START_DATE];
data = data[data.Date <= END_DATE];
new_len = len(data)
print(f"Dropped {100*(original_len - new_len)/float(original_len):.1f}% of Text Entries due to Start Date")

tqdm.pandas(desc='Calculating Word Counts')
data['wc'] = data.text.progress_apply(lambda row: len(row.split(' ')))

original_len = len(data)
data = data[data.wc < MAX_WORD_COUNT].drop('wc', axis=1)
new_len = len(data)

print(f"Dropped {100*(original_len - new_len)/float(original_len):.1f}% of Text Entries due to Word Count")

data.rename(
    columns={'ubmission': 'submissions', 'FORM_TYPEs': 'FORM_TYPE'}
    ).drop('Unnamed: 0', axis=1).to_csv(OUTPUT_NAME+'.tsv', sep='\t')

min_date = datetime(year=min(data.Date).year, month=1, day=1)
max_date = datetime(year=max(data.Date).year, month=12, day=31)

out = []
for year in range(min_date.year, max_date.year+1):
    start_date = datetime(
        year=year,
        month=1,
        day=1)
    end_date = datetime(
        year=year,
        month=12,
        day=31)

    curr_ = data
    curr_ = curr_[curr_.Date >= start_date]
    curr_ = curr_[curr_.Date <= end_date]

    print(f"{year}: {len(curr_)}")
