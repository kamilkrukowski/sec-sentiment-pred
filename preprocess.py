import pandas as pd
import numpy as np
import re

convert_label_to_string = False

f = '8k_data_labels.csv'
frame = pd.read_csv(f, delimiter='\t')

x = np.array(frame.text)
y = np.array(frame.label)

print(len(y))
valid_idx = [i for i in list(range(len(y))) if y[i] != -1]
x = x[valid_idx]
y = y[valid_idx]

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
x_test = x[test_idxs]
y_test = y[test_idxs]

with open('train.csv', 'w') as f:
    f.write('text,label\n')
    for xi, yi in zip(x_train, y_train):
        f.write(f"{xi},{yi}\n")

with open('test.csv', 'w') as f:
    f.write('text,label\n')
    for xi, yi in zip(x_test, y_test):
        f.write(f"{xi},{yi}\n")
