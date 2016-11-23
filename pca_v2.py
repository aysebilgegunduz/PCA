__author__ = 'bilge'
import numpy as np
import pandas as pd
from scipy import linalg as la
# Read data files:
train = pd.read_csv("train.csv", header=None)
test  = pd.read_csv("test.csv", header=None)


train_x = train.values[:,:-1]
#A primarily label-location based indexer, with integer position fallback.
train_y = train.ix[:,-1:]
test_x = test.values
mean = np.mean(train_x, axis=0)
mean_test = np.mean(test_x, axis=0)
meanCent = train_x - mean
meanCentTest= test_x - mean_test
#x = np.vstack(meanCent)
cov = np.cov(meanCent.T)
cov_test = np.cov(meanCentTest.T)
e_vals, e_vecs = la.eig(cov)
e_vals_test, e_vecs_test = la.eig(cov_test)
#ms = sorted(enumerate(e_vals), key=lambda x: x[1], reverse=True)
# %80 ilk 40 tanesi
print(e_vecs.shape, e_vecs_test.shape)


