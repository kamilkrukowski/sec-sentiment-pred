
from utils import prog_read_csv, subsample_yearly

RAW_DATA_NAME = '8k_data'

data = prog_read_csv(f'{RAW_DATA_NAME}.tsv', sep='\t',
                     desc='Loading 8K Data')

short = subsample_yearly(data, n=100)
short.to_csv('8k_data_short.tsv', sep='\t')