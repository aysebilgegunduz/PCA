__author__ = 'bilge'
import numpy as np
from scipy import linalg as la

A = np.random.randint(0,10,25).reshape(5,5)
e_vals, e_vecs = la.eig(A)
print(e_vals,e_vecs)