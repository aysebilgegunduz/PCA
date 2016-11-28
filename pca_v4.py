__author__ = 'bilge'
import numpy as np
import pandas as pd
from numpy import linalg as la

def euclidean(train_x, test_x, train_y, test_y):
    l = 0
    dists = []
    for k in range(len(test_x)):
        distances = (train_x - test_x[k]) ** 2
        distances = distances.sum(axis=1)
        distances = np.sqrt(distances)
        a = np.argmax(distances)
        print(train_y[a])

    return dists


# Read data files:
train = pd.read_csv("train_original.csv", header=None)
test  = pd.read_csv("test_yuzluk.csv", header=None)

perc = 0.8
train_x = train.values[:,:-1]
#train_x = train.values[:,:]
#A primarily label-location based indexer, with integer position fallback.
train_y = train.values[:,-1]
test_x = test.values[:,:-1]
test_y = test.values[:,-1]

#calculate mean
mean = np.mean(train_x, axis=0)
mean_test = np.mean(test_x, axis=0)

#adjust them
meanCent = train_x - mean
meanCentTest= test_x - mean_test

#calculate covariance matrix for both train and test
cov = np.cov(meanCent.T)
cov_test = np.cov(meanCentTest.T)

#calculate eigenvals and vecs
e_vals, e_vecs = la.eig(cov)
e_vals_test, e_vecs_test = la.eig(cov_test)

#sort all of them
#e_vals, e_vecs = sort_eigs(e_vals, e_vecs)
#e_vals_test, e_vecs_test = sort_eigs(e_vals, e_vecs)

#calculate percent of eigenvector will be chosen
limit = int(len(e_vals) * (1-perc))
limit_test = int(len(e_vals_test) * (1-perc))

#l_e_vals = e_vals[:limit]
l_e_vecs = e_vecs[:,limit:]
l_e_vecs_test = e_vecs_test[:,limit:]

print("\nFor "+str(perc)+ ": \n")
#found final data
final_data = train_x.dot(l_e_vecs)
final_data_test = test_x.dot(l_e_vecs_test)

#for i in range(len(final_data_t)):

euclidean(final_data, final_data_test, train_y, test_y)