__author__ = 'bilge'
import numpy as np
import pandas as pd
from numpy import linalg as la
from scipy.spatial import distance as ds

#eigenvalue ya gore eigenvector hesabi
def sort_eigs(e_vals, e_vecs):
    idx = e_vals.argsort()[::-1]
    eigenValues = e_vals[idx]
    eigenVectors = e_vecs[:, idx]
    return  eigenValues, eigenVectors

#euclidean distance hesabi yapiyorum
def euclidean(train_x, test_x, train_y, test_y):
    dists = []

    for k in range(test_x.shape[0]):
        distances = (train_x - test_x[k]) ** 2
        distances = distances.sum(axis=1)
        distances = np.sqrt(distances)
        a = np.argmax(distances)
        dists.append([train_y[a],test_y[k]])
    dists = np.array(dists)
    return dists
#confusion matris olusturuyor
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
#olusturulan confusion matrisi güzel sekilde ekrana basiyor
def print_conf_matrix(m):
    s = [[str(e) for e in row] for row in m]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))

#genel accuracy hesabi
def calc_accuracy(conf_matrix):
    t = sum(sum(l) for l in conf_matrix)
    return sum(conf_matrix[i][i] for i in range(len(conf_matrix))) / t

#her sinif icin accuracy hesapliyor
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
train = pd.read_csv("trainTest.csv", header=None)

perc = 0.8
train_x = train.values[:,:-1]
#A primarily label-location based indexer, with integer position fallback.
train_y = train.values[:,-1]

#calculate mean
mean = np.mean(train_x, axis=0)

#adjust them
meanCent = train_x - mean

#calculate covariance matrix for both train and test
cov = np.cov(meanCent.T)

#calculate eigenvals and vecs
e_vals, e_vecs = la.eig(cov)

#eigenvalulara gore eigenvectorleri sort ediyorum. Fonksiyonu en basta
e_vals, e_vecs = sort_eigs(e_vals, e_vecs)

#ne kadar eigenvector secilecegine burada bakiyorum
total_sum = e_vals.sum()
vecs = e_vals[0] #to calculate how many eigenvecs should've been selected
limit = 1
#perc value is temporary between 0.8, 0.6 and 0.4
while (vecs/total_sum) <= perc : #perc'in atamasini yukarıda koda baslarken yapıyorum
    vecs = vecs + e_vals[limit]
    limit += 1
limit += 1 #0dan basladigi icin

l_e_vecs = e_vecs[:,:limit] #secilen sutunlari yeni bir matrise atiyorum

print("\nFor "+str(perc)+ ": \n")

#calculated final data
final_data = meanCent.dot(l_e_vecs) #matris carpımı yapiliyor burada

#euclidean distance hesabindan gelen dists = [predicted, actual] seklinde bir data
dists = euclidean(final_data[:50,:],final_data[50:150,:],train_y[:50], train_y[50:150])
#confusion matris olusuyor
m = confusion_matrix(dists)
#her sinif icin accuracy hesabi
calc_accuracy_each_matrix(m)
