# -*- coding: utf-8 -*-
"""
Introduction to Markov Chains
Code for Markov Chain Exercises

MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay
@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import numpy as np
from numpy.linalg import eig

# Transition Matrix
T = np.array([[  0, 0.8, 0.9, 0.4],
              [0.6,   0,   0, 0],
              [  0, 0.9,   0, 0],
			  [  0,   0, 0.8, 0]])
			  
# Check if Matrix T is a Markov

# Sum each row of matrix T
print('\nSum of Columns from Matrix T')
print(T.sum(axis=0))

# Compute Eigen Values and Vectors for Transition 

w,v = eig(T)
# Eigenvectors
# This gives normalized (unit “length”) eigenvectors, such that the column 
#   v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
print('\nE-values:', w)
print('E-vectors:\n', v, '\n')
l0 = w[0]
v0 = v[:,0]
print('Dominant Eigenvalue:',l0)
print('Associated Eigenvector Normalized:\n',v0/np.sum(v0))

# %% [3] Example 3


T3 = np.array([[0.8, 0.1],
              [0.2, 0.9]])

w,v = eig(T3)
print('\nE-values:', w)
print('E-vectors:\n', v, '\n')
l1 = w[1]
v1 = v[:,1]
print('Eigenvalue:',l1)
print('Associated Eigenvector Normalized:\n',v1/np.sum(v1))

# %% [4] Exercise 4

T4 = np.array([[0.5, 0.4, 0.1],
              [0.4, 0.2, 0.3],
              [0.1, 0.4, 0.6]])

X0 = np.array([[80],
               [40],
               [80]])

n = 3
Xin = X0
for i in range(n):
    Xout = np.dot(T4,Xin)
    Xin = Xout
    
print('\nFor n =', n, 'and Xout =\n', Xout)
print('\nHere is the normalized version of the vector\n', Xout/np.sum(Xout))

w,v = eig(T4)
print('\nE-values:', w)
print('E-vectors:\n', v, '\n')
l3 = w[2]
v3 = v[:,2]
print('Eigenvalue:',l1)
print('Associated Eigenvector Normalized:\n',v3/np.sum(v3))



