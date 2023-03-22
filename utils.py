from math import ceil
from tqdm.auto import tqdm

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
