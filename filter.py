from math import ceil
from datetime import datetime
import os

from tqdm.auto import tqdm
import pandas as pd
import numpy as np

RAW_DATA_NAME = '8k_data_raw_long.tsv'
OUTPUT_NAME = '8k_data_filtered.tsv'
START_DATE = datetime.strptime('2010-01-01', '%Y-%m-%d')
END_DATE = datetime.strptime('2022-12-31', '%Y-%m-%d')
N_MOVING_AVG = 7
MAX_WORD_COUNT = 8000

# Estimate number of lines
n_lines = 0
with open(RAW_DATA_NAME, 'r') as f:
    n_lines = sum([1 for _ in f])

# Prepare file output for streamed appending
if os.path.exists(OUTPUT_NAME):
    os.system(f'rm {OUTPUT_NAME}')
os.system(f'touch {OUTPUT_NAME}')

# We read 10k lines per iteration
chunk = 5000.0
total_reads = ceil(n_lines / chunk)

# Set up tqdm iterable
desc = "Filtering Entries"
itera = tqdm(pd.read_csv(RAW_DATA_NAME, sep='\t', chunksize=chunk),
             total=total_reads, desc=desc)

# Track metrics for filtering
total_length = 0
dropped_length = 0
samples_by_year = dict()

# Write header ONCE
header = True

for curr in itera:

    # Process DateTime
    if 'Date' in curr.columns:
        elem_ = curr.Date.iloc[0]
        if isinstance(elem_, float):
            elem_ = str(elem_)
        elif isinstance(elem_, int):
            elem_ = str(elem_)
        elif isinstance(elem_, np.int64):
            elem_ = str(elem_)
        if isinstance(elem_, str):
            if len(elem_) == 10:
                curr['Date'] = pd.to_datetime(curr['Date'], format="%Y-%m-%d")
            elif len(elem_) == 8:
                curr['Date'] = pd.to_datetime(curr['Date'], format="%Y%m%d")

    # Filter on dates
    start_length = len(curr)
    curr = curr[curr.Date >= START_DATE];
    curr = curr[curr.Date <= END_DATE];

    # Filter on max word count
    curr['wc'] = curr.text.apply(lambda row: len(row.split(' ')))
    curr = curr[curr.wc < MAX_WORD_COUNT].drop('wc', axis=1)
    new_len = len(curr)

    # Track metrics
    dropped_length += (start_length - new_len)
    total_length += start_length

    if new_len == 0:
        continue

    # Track numbers of samples by year
    min_date = datetime(year=min(curr.Date).year, month=1, day=1)
    max_date = datetime(year=max(curr.Date).year, month=12, day=31)
    for year in range(min_date.year, max_date.year+1):
        start_date = datetime(
            year=year,
            month=1,
            day=1)
        end_date = datetime(
            year=year,
            month=12,
            day=31)

        data_ = curr
        data_ = data_[data_.Date >= start_date]
        data_ = data_[data_.Date <= end_date]

        if year not in samples_by_year:
            samples_by_year[year] = 0
        samples_by_year[year] += len(data_)
        del data_

    # Append only
    curr.rename(
        columns={'ubmission': 'submission', 'FORM_TYPEs': 'FORM_TYPE'}
        ).drop('Unnamed: 0', axis=1).to_csv(
            OUTPUT_NAME, sep='\t', mode='a', header=header, index=False)

    if header:
        header = False

print(f"Dropped {100*(total_length - dropped_length)/float(total_length):.1f}% of Text Entries")
print(samples_by_year)
