from math import ceil
from tqdm.auto import tqdm
from typing import List, Any
from datetime import datetime, timedelta


import pandas as pd
import numpy as np

from metrics import Metrics

def prog_read_csv(path, **read_params):
    """Pandas.read_csv with tqdm loading bar"""

    # Estimate number of lines/chunks if not present in kwargs
    n_lines = 0
    if 'nrows' in read_params and read_params['nrows'] is not None:
        n_lines = read_params['nrows']
    else:
        with open(path, 'r') as f:
            n_lines = len(f.readlines())

    total_reads = None
    if 'chunksize' not in read_params or read_params['chunksize'] < 1:
        if 'nrows' in read_params and read_params['nrows'] is not None:
            n_lines = read_params['nrows']
        chunk = max(ceil(n_lines / 100.0), 100)
        chunk = min(chunk, n_lines)
        total_reads = ceil(n_lines / chunk)
        read_params['chunksize'] = chunk

    # Set up tqdm iterable
    desc = read_params.pop('desc', None)
    itera = tqdm(pd.read_csv(path, **read_params), total=total_reads,
                 desc=desc)

    # Read chunks and re-assemble into frame
    out = pd.concat(itera, axis=0)

    if 'Date' in out.columns:
        elem_ = out.Date.iloc[0]
        if isinstance(elem_, float):
            elem_ = str(elem_)
        elif isinstance(elem_, int):
            elem_ = str(elem_)
        if isinstance(elem_, str):
            if len(elem_) == 10:
                out['Date'] = pd.to_datetime(out['Date'], format="%Y-%m-%d")
            elif len(elem_) == 8:
                out['Date'] = pd.to_datetime(out['Date'], format="%Y%m%d")

    return out


def datasplit_dframe(frame, split=[0.8, 0.2]):
    """Conduct a train/test/dev split on a dataframe after shuffle."""
    perm = np.random.permutation(len(frame))

    lens = []
    total = len(frame)
    for perc in split[:-1]:
        lens.append(int(perc * total))

    idxs = []
    for start, stop in zip([None] + lens, lens + [None]):
        idxs.append(perm[start:stop])

    output = []
    for idxs_ in idxs:
        output.append(frame.iloc[idxs_])
    return tuple(output)


def get_tikr_indices_file(tikr: str, dfpath,
                          progbar=True, **kwargs) -> List[int]:
    """Return what rows in DF correspond to tikr in column tikr."""
    df = None
    if progbar:
        df = prog_read_csv(dfpath, usecols=['tikr'],
                           desc='Getting TIKR indices', **kwargs)
    else:
        df = pd.read_csv(dfpath, usecols=['tikr'], **kwargs)

    mask = np.array(df) == tikr
    idxs = np.nonzero(mask)[0]
    return idxs


class ChronoYearly():
    """Cross-validation data split generator. Generates 1-year period test set splits."""

    def __init__(self, df, period_length=365, periods_to_test=5, verbose=True):

        self._min_date = datetime(year=min(df.Date).year, month=1, day=1)
        self._max_date = datetime(year=max(df.Date).year, month=12, day=31)
        number_periods = self._max_date.year - self._min_date.year + 1

        assert periods_to_test < number_periods, 'Not enough historical data'

        if verbose:
            print(f"Across {number_periods} years, "
                  f"train on {number_periods-periods_to_test} "
                  f"with 1-year duration testing on {periods_to_test} "
                  "different years")

        self.df = df
        self._idx = 0
        self.periods_to_test = periods_to_test - 1
        self.period_length = period_length

    def __iter__(self):
        self.idx_ = 0
        return self

    def __next__(self):
        if self._idx == self.periods_to_test:
            self._idx = 0
            raise StopIteration
        start_date = datetime(
            year=self._min_date.year +
            self._idx,
            month=1,
            day=1)
        end_date = datetime(
            year=self._max_date.year -
            self.periods_to_test + self._idx + 1,
            month=12,
            day=31)
        generate_data_splits = end_date - timedelta(self.period_length)

        out = self.df
        out = out[out.Date >= start_date]
        out = out[out.Date <= end_date]

        boundary = generate_data_splits
        train = out[out.Date <= boundary]

        val_test = out[out.Date > boundary]
        val_test = val_test.sort_values(by='Date',ascending=True)

        val = val_test[val_test['Date'] < datetime(year=min(val_test.Date).year, month=4, day=1)]
        test = val_test[val_test['Date'] >= datetime(year=min(val_test.Date).year, month=4, day=1)]
    
        self._idx += 1
        
        return train, val, test




def generate_data_splits(df, strategy='chronological_yearly', periods_to_test=5,
                         period_length=365, verbose=True):

    if strategy == 'chronological_yearly':
        return ChronoYearly(
            df, period_length=period_length, periods_to_test=periods_to_test, verbose=verbose)

def subsample_yearly(df, n=1000):
    min_date = datetime(year=min(df.Date).year, month=1, day=1)
    max_date = datetime(year=max(df.Date).year, month=12, day=31)

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

        curr_ = df
        curr_ = curr_[curr_.Date >= start_date]
        curr_ = curr_[curr_.Date <= end_date]

        n_ = min(len(curr_), n)
        out.append(curr_.sample(n_).copy())

    return pd.concat(out, axis=0)