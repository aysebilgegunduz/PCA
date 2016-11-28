__author__ = 'bilge'
import numpy as np
import pandas as pd
from numpy import linalg as la

def sort_eigs(e_vals, e_vecs):
    idx = e_vals.argsort()[::-1]
    eigenValues = e_vals[idx]
    eigenVectors = e_vecs[:, idx]
    return  eigenValues, eigenVectors


def euclidean(train_x, test_x, train_y, test_y):
    l = 0
    dists = []
    for k in range(test_x.shape[0]):
        distances = (train_x - test_x[k]) ** 2
        distances = distances.sum(axis=1)
        distances = np.sqrt(distances)
        a = np.argmax(distances)
        dists.append([train_y[a],test_y[k]])
    dists = np.array(dists)

    return dists

def confusion_matrix(dists):
    actual = dists[:,-1:]
    predicted = dists[:,:1]
    n_classes = 10
    m = [[0] * n_classes for i in range(n_classes)]
    for act, pred in zip(actual, predicted):
        m[act][pred] += 1
    print("Accuracy: ",calc_accuracy(m))
    print("\nConfusion Matrix [actual][predicted]:\n")
    print_conf_matrix(m)
    return m

def print_conf_matrix(m):
    s = [[str(e) for e in row] for row in m]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t

def calc_accuracy_each_matrix(m):
    print("\nAccuracy for each classes: \n")
    n_classes=10 #number of classes
    k=0
    for i in m:
        if k < n_classes:
            tp = i[k]
            total = sum(l for l in i)
            print("Class "+str(k)+": ", tp/total)
            k += 1
    return None

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
e_vals, e_vecs = sort_eigs(e_vals, e_vecs)
e_vals_test, e_vecs_test = sort_eigs(e_vals, e_vecs)

#calculate percent of eigenvector will be chosen
limit = int(len(e_vals) * (1-perc))
limit_test = int(len(e_vals_test) * (1-perc))

#l_e_vals = e_vals[:limit]
l_e_vecs = e_vecs[:,limit:]
l_e_vecs_test = e_vecs_test[:,limit:]

print("\nFor "+str(perc)+ ": \n")
#found final data
#final_data = np.dot(l_e_vecs.T, meanCent.T)
#final_data_test = np.dot(l_e_vecs_test.T, meanCentTest.T)
final_data = meanCent.dot(l_e_vecs)
final_data_test = meanCentTest.dot(l_e_vecs_test)
#calculation started
dists = euclidean(final_data,final_data_test,train_y,test_y)
m = confusion_matrix(dists)
calc_accuracy_each_matrix(m)
#true_positives_perc(dists)
#print(tp)
#print(np.matrix(final_data).T.shape, (np.matrix(final_data_test).T.shape))