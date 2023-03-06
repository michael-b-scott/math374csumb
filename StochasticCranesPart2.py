# -*- coding: utf-8 -*-
"""
Stochastics Cranes Part 2 - Eigenvector and Eigenvalue Analysis
MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay

@author: Michael B. Scott
@email: mscott@csumb.edu
"""
# %% [0] Load Libraries and Data

# Libraries
import pandas as pd
import numpy as np
from numpy.linalg import eig
import matplotlib.pyplot as plt

# Load Sandhill Crane Demographic Data
#   Survival and Reproduction Rates Best and Worst Cases
file = 'https://raw.githubusercontent.com/michael-b-scott/math374csumb/main/sandhill-demographics.csv'
demodata = pd.read_csv(file)

# -----------------------------------------------------------------------------
# %% [1] Create Transition Matrix from Demographic Data
#   Here we choose Best Case for Survival and Reproduction
# -----------------------------------------------------------------------------

# Remove first row - Use chosen initial values
demodata = demodata.iloc[1:]

# Initial Row - Captures Reproduction Rates
x1 = demodata['rBest'].T

# Create Transition Matrix A
A = np.array([x1])

# Add Remaining Rows (Row>1) - Captures Survival Rates for Given Year
row0 = np.zeros((1, len(x1)))
for i in range(1, len(demodata['sBest'])):
    print(i)
    thisrow = row0
    thisrow[0,i-1] = demodata['sBest'][i]
    A = np.vstack((A,thisrow))
    thisrow[0,i-1] = 0.0

# -----------------------------------------------------------------------------
# %% [2] Recreate Table 3.7 from Modeling Text for 21 years

# Choose Initial Population Distribution and Create Demographic Data Table
x0 = np.full((20,1), 2.0, dtype=np.float64)

craneDemoData = pd.DataFrame().assign(Age_Class=demodata['Age Class'], Birth_Rate=demodata['rBest'], Survival_Rate=demodata['sBest'])
craneDemoData = craneDemoData.assign(t=" ")
craneDemoData = craneDemoData.reset_index(drop=True)
x0r = pd.DataFrame(x0)
craneDemoData = pd.concat([craneDemoData, x0r], axis=1)

# Number of years in projection
# You change this for the class exercises
years = 36


cp = x0
cptotal = pd.DataFrame({'Total': pd.Series(dtype='float')})
cptotal.loc[len(cptotal.index)] = x0.sum(axis=0)
cpgrowth = pd.DataFrame({'Growth Rate': pd.Series(dtype='float')})
cpgrowth.loc[len(cpgrowth.index)] = ''
for i in range(1,years+1):
    cpm1 = cp 
    cpm1sum = cpm1.sum()
    cp = np.dot(A,cp)
    cpsum = cp.sum()
    thiscolumn = pd.DataFrame(cp, columns=[str(i)])
    
    # Get Crane Pop Total and Growth Rate by Year
    cptotal.loc[len(cptotal.index)] = cpsum # thiscolumn[str(i)].sum(axis=0)
    cpgrowth.loc[len(cpgrowth.index)] = round(100*(cpsum-cpm1sum)/cpm1sum, 2)
    thiscolumn = thiscolumn.round(decimals=2)
    craneDemoData = pd.concat([craneDemoData, thiscolumn], axis=1)

# Get Total by Age Class Across All Years
cptotal = cptotal.round(2)

years_array = pd.DataFrame(range(years+1), columns=['Year'])

craneDemoDataTotals = pd.concat([years_array, cptotal, cpgrowth], axis=1)

# -----------------------------------------------------------------------------
# %% [3] Line Graph over 20 years


fig, ax = plt.subplots(figsize=(5, 4), layout='constrained')
ax.plot(years_array,cptotal, c="blue", marker='v', label='Sandhill Crane Total Pop')


ax.set_xlabel('Years')  # Add an x-label to the axes.
ax.set_ylabel('Population Size')  # Add a y-label to the axes.
ax.set_title("Sandhill Crane Projections %i Years" %years)  # Add a title to the axes.
#plt.xlim([0, 155])
#plt.ylim([0, 60])
ax.grid(True)
#ax.legend();  # Add a legend.


# -----------------------------------------------------------------------------
# %% [x] Compute Eigenvalues and Eigenvectors of Transition Matrix

w,v = eig(A)
# Eigenvectors
# This gives normalized (unit “length”) eigenvectors, such that the column 
#   v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
print('\nE-values:\n', w)
#print('\nE-vectors:\n', v, '\n')
l0 = w[0]
v0 = v[:,0]
# Dominant Eigenvector Normalized
v0n = pd.Series(np.real(v0/np.sum(v0)))
# Population age distribution for final year.
pFinalYear = craneDemoData[str(years)]/craneDemoData[str(years)].sum() 
#Difference final year vs. dominant eigenvector
diffFYwv0 = pFinalYear-v0n
# Print Findings
print('Dominant Eigenvalue:',l0)
print('Associated Eigenvector Normalized:\n',v0n)
print('\nCompare normalized eigenvector of dominant eigenvalue with normalized ')
print('vector of the final year of projection, year', years,'.\n')
print('\nFinal year age distribution\n')
print(pFinalYear)
print('\nDifference final year vs. dominant eigenvector\n')
print(diffFYwv0.round(4))
