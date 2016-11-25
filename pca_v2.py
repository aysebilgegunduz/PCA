__author__ = 'bilge'
import numpy as np
import pandas as pd
from scipy import linalg as la

def euclidean(train, test):
    return None
# Read data files:
train = pd.read_csv("train_original.csv", header=None)
test  = pd.read_csv("test.csv", header=None)

perc = 0.8
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
limit = len(e_vals) * (1-perc)
limit_test = len(e_vals_test) * (1-perc)
l_e_vecs = e_vecs[:,limit:]
l_e_vecs_test = e_vecs_test[:,limit:]
final_data = np.dot(l_e_vecs.T, meanCent.T)
final_data_test = np.dot(l_e_vecs_test.T, meanCent.T)
#e_vals_test, e_vecs_test = la.eig(cov_test)
#ms = sorted(enumerate(e_vals), key=lambda x: x[1], reverse=True)
# %80 ilk 40 tanesi
print(np.matrix(final_data).T.shape, (np.matrix(final_data_test).T.shape))