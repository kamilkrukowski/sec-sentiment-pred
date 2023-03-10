from typing import List, Dict
from datetime import datetime
from typing import List, Dict, Any, Union, Tuple


import numpy as np

Date = Union[datetime, str]
label = int


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


def uniform(
    predictions: List[Tuple[Date, label]],
    company_list: List[str]
) -> Dict[datetime, List[float]]:
    # Calculate portfolio holdings at time intervals given 8-K labels.

    dates = []
    labels = []
    for date, label in predictions:
        dates.append(date)
        labels.append(label)
    dates, labels = chronosort(dates, labels)

    uniform = [1.0/len(company_list)] * len(company_list)

    end_year = dates[-1].year

    out = {dates[0]: uniform}

    curr_year = dates[0].year + 1
    while curr_year < end_year:
        out[datetime.strptime(f"{curr_year}0101", "%Y%m%d")] = uniform
        curr_year += 1

    return out
