# -*- coding: utf-8 -*-
"""
Empirical Modeling Part 2
Correlation and Fitting a Line to Data Using the 
Least-Squares Criterion - Linear Regression


MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay
@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import pandas as pd
import statsmodels.api as sm
import matplotlib.pyplot as plt

# Black Spruce Seedlings
'''
Black spruce (Picea mariana) is a species of a slow-growing coniferous tree found
across the northern part of North America. It is commonly found on wet organic
soils. In a study conducted in the 1990s, a biologist interested in factors affecting the
growth of the black spruce planted its seedlings on sites located in boreal peatlands
in northern Manitoba, Canada (Camill et al. (2010)).

The data set Spruce contains a part of the data from the study.
Seventy-two black spruce seedlings were planted in four plots under varying conditions
(fertilizer–no fertilizer, competition–no competition) and their heights and
diameters were measured over the course of 5 years.

The researcher wanted to see whether the addition of fertilizer or the removal of
competition from other plants (by weeding) affected the growth of these seedlings.
'''

# -----------------------------------------------------------------------------
# Load data
file = "https://raw.githubusercontent.com/michael-b-scott/math374csumb/main/Spruce.csv"
data = pd.read_csv(file)

dh = data['Ht.change']
dd = data['Di.change']

# -----------------------------------------------------------------------------
# Scatter Plot H vs D with quadrants by mean values
fig, ax = plt.subplots()
ax.scatter(dh, dd, s=30, facecolor='none', ec="black")
ax.set_xlabel('Change in height (cm)')  # Add an x-label to the axes.
ax.set_ylabel('Change in diameter (cm)')  # Add a y-label to the axes.
#plt.ylim([4,13])
ax.set_title("Change in height vs. diameter for Black Spruce")  # Add a title to the axes.

ax.text(32, 8.5, 'I')
ax.text(29, 8.5, 'II')
ax.text(29, 1, 'III')
ax.text(32, 1, 'IV')

plt.axhline(y=dd.mean(), color='black')
plt.axvline(x=dh.mean(), color='black')
#ax.legend();  # Add a legend.

# -----------------------------------------------------------------------------
# Compute covariance and correlation coefficient (Using Pandas)
# Note: We compute covariance  of two Pandas series
print('\nCovariance of change in height and change in diameter with Pandas\n')
print(dh.cov(dd))

# Compute correlation
print('\nCorrelation between change in height and change in diameter with Pandas\n')
print(dh.corr(dd))

# Check formula
print('\nCorrelation computed from covariance and standard deviation\n')
print(dh.cov(dd)/(dh.std()*dd.std()))


# -----------------------------------------------------------------------------
# Linear Regression

# Model needs an intercept, so we add a column of 1s
dhc = sm.add_constant(dh)
lm = sm.OLS(dd, dhc)
result = lm.fit()
print(result.summary())

fig2, ax2 = plt.subplots(figsize=(8, 6))

ax2.scatter(dh, dd, facecolor='none', ec="black", label="Data")
ax2.plot(dh, result.fittedvalues, color='black', label="OLS")
x = [dh[49], dh[49]]
y = [result.fittedvalues[49], dd[49]]
ax2.plot(x, y, ls="--", color='black')
ax2.plot(dh[49] ,result.fittedvalues[49], marker="o", markeredgecolor="black", markerfacecolor="black")
ax2.set_xlabel('Change in height (cm)')  # Add an x-label to the axes.
ax2.set_ylabel('Change in diameter (cm)')  # Add a y-label to the axes.
ax2.text(dh[49]-5, dd[49], r'$(x,y)$', fontsize=14,)
ax2.text(dh[49]+1.5, y[0]-0.1, r'$(x,\beta_0+\beta_1 x)$', fontsize=14,)

