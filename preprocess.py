from utils import prog_read_csv, datasplit_dframe


f = '8k_data_labels.tsv'
frame = prog_read_csv(f, delimiter='\t', desc='Loading Data...')
keys = ['text', 'label', 'Date']

train_df, test_df = datasplit_dframe(frame[keys], [0.8, 0.2])

train_df.to_csv('train.tsv', sep='\t')
test_df.to_csv('train.tsv', sep='\t')
