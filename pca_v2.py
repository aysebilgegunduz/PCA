__author__ = 'bilge'
import numpy as np
import pandas as pd
from scipy import linalg as la
# Read data files:
train = pd.read_csv("train.csv")
test  = pd.read_csv("test.csv")


train_x = train.values[:,:-1]
#A primarily label-location based indexer, with integer position fallback.
train_y = train.ix[:,-1:]
test_x = test.values
mean = np.mean(train_x, axis=0)
#x = np.vstack([a,b,c])
#cov = np.cov(x)
#e_vals, e_vecs = la.eig(A)
#print(e_vals,e_vecs)
print(mean)
