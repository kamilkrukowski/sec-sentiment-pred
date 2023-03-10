import pandas as pd
import numpy as np
import re

convert_label_to_string = False

f = '8k_data_labels.tsv'
frame = pd.read_csv(f, delimiter='\t')

x = np.array(frame.text)
y = np.array(frame.label)
dates = np.array(frame.Date)

print(len(y))
valid_idx = [i for i in list(range(len(y))) if y[i] != -1]
x = x[valid_idx]
y = y[valid_idx]
dates = dates[valid_idx]

if convert_label_to_string:
    id2label = {0: 'negative', 1: 'positive', 2: 'neutral'}
    temp_y = list()
    for i in range(len(y)):
        temp_y.append(id2label[y[i]])
    y = np.array(temp_y)

print(f"There are {len(y)} valid samples")
print(f"The unique prediction classes: {np.unique(y)}")

idxs = np.random.permutation(len(y))
len_train = int(0.8*len(y))

train_idxs = idxs[:len_train]
test_idxs = idxs[len_train:]

x_train = x[train_idxs]
y_train = y[train_idxs]
dates_train = dates[train_idxs]
x_test = x[test_idxs]
y_test = y[test_idxs]
dates_test = dates[test_idxs]

with open('train.tsv', 'w') as f:
    f.write('text\tlabel\tDate\n')
    for xi, yi, di in zip(x_train, y_train, dates_train):
        f.write(f"{xi}\t{yi}\t{di}\n")

with open('test.tsv', 'w') as f:
    f.write('text\tlabel\tDate\n')
    for xi, yi, di in zip(x_test, y_test, dates_test):
        f.write(f"{xi}\t{yi}\t{di}\n")
