"""
Empirical Modeling Part 8: Curve Fitting Numpy Polyfit



MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay
@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import matplotlib.pyplot as plt
import numpy as np
from numpy.polynomial import polynomial as P
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit


#Data Source
#import yfinance as yf

#data = yf.download(tickers="DJIA", period="1m", interval="30y")

# -----------------------------------------------------------------------------
# Polyfit and Polyval Introduction

# Generate Data
# importing random module
import random
random.seed(0)

L = 21
N = L
STDEV = 10
x = np.linspace(-5, 5, N)
y_true = P.polyval(x, [0,-1,0,1])
y = y_true + np.random.normal(0, STDEV, N)

ypolyfit3d = P.polyval(x, P.polyfit(x,y,3))
ypolyfit3d = pd.Series(ypolyfit3d)
ypolyfit4d = P.polyval(x, P.polyfit(x,y,4))
ypolyfit4d = pd.Series(ypolyfit4d)



print('\nPolynomial Odd Deg Model Deg 3 R-Squared = ', pow(ypolyfit3d.corr(pd.Series(y)),2))
print('\nPolynomial Odd Deg Model Deg 4 R-Squared = ', pow(ypolyfit4d.corr(pd.Series(y)),2))


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.plot(x, ypolyfit3d, color='green', linestyle='dotted', label="Polyfit Deg 3")
ax.plot(x, ypolyfit4d, color='blue', linestyle='dashdot', label="Polyfit Deg 4")
ax.plot(x, -x + x**3, color='black', label='Actual', linestyle='dashed')
ax.set_title('Polynomial Odd Degree')
ax.set_xlabel('x')  # Add an x-label to the axes.
ax.set_ylabel('y')  # Add a y-label to the axes.
ax.grid()
ax.legend()


# Polyfit with even degree
y_true = P.polyval(x, [-10,1,-0.5,0,0.1])
y = y_true + np.random.normal(0, STDEV, N)

ypolyfit3d = P.polyval(x, P.polyfit(x,y,3))
ypolyfit3d = pd.Series(ypolyfit3d)
ypolyfit4d = P.polyval(x, P.polyfit(x,y,4))
ypolyfit4d = pd.Series(ypolyfit4d)

print('\nPolynomial Even Deg Model Deg 3 R-Squared = ', pow(ypolyfit3d.corr(pd.Series(y)),2))
print('\nPolynomial Even Deg Model Deg 4 R-Squared = ', pow(ypolyfit4d.corr(pd.Series(y)),2))

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.plot(x, ypolyfit3d, color='green', linestyle='dotted', label="Polyfit Deg 3")
ax.plot(x, ypolyfit4d, color='blue', linestyle='dashdot', label="Polyfit Deg 4")
ax.plot(x, y_true, color='black', label='Actual', linestyle='dashed')
ax.set_title('Polynomial Even Degree')
ax.set_xlabel('x')  # Add an x-label to the axes.
ax.set_ylabel('y')  # Add a y-label to the axes.
ax.grid()
ax.legend()
