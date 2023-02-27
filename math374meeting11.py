# -*- coding: utf-8 -*-
"""
Introduction to Matrix Algebra in Python
MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay

@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

# -----------------------------------------------------------------------------
# Human Population Model Section 3.1 in text
# -----------------------------------------------------------------------------

# Birth rates
b1 = 0.4271 # Age 0-15
b2 = 0.8498 # Age 15-30
b3 = 0.1273 # Age 30-45

# Survival Rates
s12 = 0.9924 # Age 0-15
s23 = 0.9826 # Age 15-30

# Initial Populations by Age Group in Millions
x1 = 30 # Age 0-15
x2 = 30 # Age 15-30
x3 = 30 # Age 30-45

# Recursion Matrix
A = np.array([[b1, b2, b3],
              [s12, 0,  0],
              [ 0, s23, 0]])

p0 = np.array([[x1],
               [x2],
               [x3]])


hpdata = pd.DataFrame({'Year': pd.Series(dtype='int'),
                       'Age 0-15': pd.Series(dtype='float'),
                       'Age 15-30': pd.Series(dtype='float'),
                       'Age 30-45': pd.Series(dtype='float'),
                       'Total': pd.Series(dtype='float')})
# Initial Populations
hpdata.loc[len(hpdata.index)] = [0, x1, x2, x2,np.sum(p0)]

# Population Distribution Among Age Classes
hpdist = pd.DataFrame({'Step': pd.Series(dtype='int'),
                       'Age 0-15': pd.Series(dtype='float'),
                       'Age 15-30': pd.Series(dtype='float'),
                       'Age 30-45': pd.Series(dtype='float')})
sum0 = x1 + x2 + x3
hpdist.loc[len(hpdist.index)] = [0, x1/sum0, x2/sum0, x2/sum0]

for i in range(1, 11):
    # Population Data
    prepop = hpdata.iloc[i-1,1:4] 
    pop = np.dot(A,prepop)
    poptot  = np.sum(pop)
    hpdata.loc[len(hpdata.index)] = [i*15, pop[0], pop[1], pop[2], poptot]
    
    popdist = pop/poptot
    hpdist.loc[len(hpdist.index)] = [i, popdist[0], popdist[1], popdist[2]] 
     

# Plot Output
fig, ax = plt.subplots(figsize=(8, 4), layout='constrained')
#ax.scatter(x=hpdata['Year'], y=hpdata['Age 0-15'], marker="v", s=70, label='Age 0-15', c="blue")
ax.plot(hpdata['Year'],hpdata['Age 0-15'], c="blue", marker='v', label='Age 0-15')
ax.plot(hpdata['Year'],hpdata['Age 15-30'], c="green", marker='s', label='Age 15-30')
ax.plot(hpdata['Year'],hpdata['Age 30-45'], c='tab:brown', marker='^', label='Age 30-45')

ax.set_xlabel('Years')  # Add an x-label to the axes.
ax.set_ylabel('Population (Millions)')  # Add a y-label to the axes.
ax.set_title("Human Female Population Model")  # Add a title to the axes.
#plt.xlim([0, 155])
plt.ylim([0, 250])
ax.grid(True)
ax.legend();  # Add a legend.

# Compute Eigen Values and Vectors for Transition 

w,v = eig(A)
# Eigenvectors
# This gives normalized (unit “length”) eigenvectors, such that the column 
#   v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
print('\nE-values:', w)
print('E-vectors:\n', v, '\n')
l0 = w[0]
l1 = w[1]
l3 = w[2]
v0 = v[:,0]
v1 = v[:,1]
v2 = v[:,2]
print('Dominant Eigenvalue:',l0)
print('Associated Eigenvector Normalized:\n',v0/np.sum(v0))

