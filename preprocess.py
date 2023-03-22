from utils import prog_read_csv, datasplit_dframe


f = '8k_data.tsv'
keys = ['text', 'label', 'Date', 'annual return', 'percent return']
frame = prog_read_csv(f, delimiter='\t', usecols=keys, desc='Loading Data...')
print(f"Length: {len(frame)}")

train_df, test_df = datasplit_dframe(frame[keys], [0.8, 0.2])

train_df.to_csv('train.tsv', sep='\t')
test_df.to_csv('test.tsv', sep='\t')
