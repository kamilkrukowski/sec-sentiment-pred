import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score

from utils import prog_read_csv

keys = ['label', 'annual return', 'percent return']
dftrain = prog_read_csv('train.tsv', sep='\t', usecols=keys,
                        desc='Loading Training Data')  # load cleaned data
dftest = prog_read_csv('test.tsv', sep='\t', usecols=keys,
                       desc='Loading Test Data')  # load cleaned data

# x feature, c class, y price
c_train = np.array(dftrain['label'])
c_test = np.array(dftest['label'])
ac_train = np.array(dftrain['annual return'])
ac_test = np.array(dftest['annual return'])
pc_train = np.array(dftrain['percent return'])
pc_test = np.array(dftest['percent return'])

plt.hist(ac_train[c_train == 1], bins=np.array(range(40))/10-1, alpha=0.6)
plt.hist(ac_train[c_train == 0], bins=np.array(range(40))/10-1, alpha=0.6)
plt.savefig('fig.png')
plt.clf()

ay = ac_train[c_train != 2]
py = pc_train[c_train != 2] 
c = c_train[c_train != 2]

print("\nAbility to distinguish Sentiment Class by using Return as Score"
      f"\nAUROC: {roc_auc_score(c, ay):.3f}")
print('\nMedian Annualized Return by Sentiment')
print(f"Any: {np.median(ay):.3f}")
print(f"Negative: {np.median(ay[c==0]):.3f}")
print(f"Positive: {np.median(ay[c==1]):.3f}")
print('\nMedian Percent Return by Sentiment')
print(f"Any: {np.median(py):.3f}")
print(f"Negative: {np.median(py[c==0]):.3f}")
print(f"Positive: {np.median(py[c==1]):.3f}")
