from math import ceil
from tqdm.auto import tqdm
from typing import List, Any
from datetime import datetime


import pandas as pd
import numpy as np


def prog_read_csv(path, **read_params):
    """Pandas.read_csv with tqdm loading bar"""

    # Estimate number of lines/chunks
    n_lines = 0
    with open(path, 'r') as f:
        n_lines = len(f.readlines())
    if 'chunksize' not in read_params or read_params['chunksize'] < 1:
        read_params['chunksize'] = max(ceil(n_lines/100.0), 100)

    # Set up tqdm iterable
    desc = read_params.pop('desc', None)
    itera = tqdm(pd.read_csv(path, **read_params), total=100,
                 desc=desc)

    # Read chunks and re-assemble into frame
    return pd.concat(itera, axis=0)

def datasplit_dframe(frame, split=[0.8, 0.2]):
    """Conduct a train/test/dev split on a dataframe after shuffle."""
    perm = np.random.permutation(len(frame))

    lens = []
    total = len(frame)
    for perc in split[:-1]:
        lens.append(int(perc*total))

    idxs = []
    for start, stop in zip([None] + lens, lens + [None]):
        idxs.append(perm[start:stop])

    output = []
    for idxs_ in idxs:
        output.append(frame.iloc[idxs_])
    return tuple(output)

def chronosort(dates: List[datetime], arr: List[Any],
               start_date: datetime = None, end_date: datetime = None):
    """Sort dates and arr in-place by dates, remove all
    entries before start_date or after end_date."""

    if type(start_date) is str:
        start_date = datetime.strptime(start_date, '%Y%m%d')

    if type(end_date) is str:
        end_date = datetime.strptime(end_date, '%Y%m%d')

    dates = np.array(dates)
    arr = np.array(arr)

    sort_idxs = np.argsort(dates)
    arr = arr[sort_idxs]
    dates = dates[sort_idxs]

    if start_date is not None or end_date is not None:
        dates = list(dates)
        arr = list(arr)

        if start_date is not None:
            dates.insert(0, start_date)
            arr.insert(0, -999)
        elif end_date is not None:
            dates.insert(1, end_date)
            arr.insert(1, -999)

    sort_idxs = list(np.argsort(dates))

    start_idx = None
    end_idx = None

    if start_date is not None:
        start_idx = sort_idxs.index(0) + 1  # We exclude our dummy entry

    if end_date is not None:
        end_idx = sort_idxs.index(1)

    dates = np.array(dates)
    arr = np.array(arr)
    dates = dates[sort_idxs][start_idx:end_idx]
    arr = arr[sort_idxs][start_idx:end_idx]

    return dates, arr