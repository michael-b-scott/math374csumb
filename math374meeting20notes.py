"""
Empirical Modeling Part 6: Curve Fitting with Power and Exponential Functions
Excercise 1


MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay
@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import matplotlib.pyplot as plt
#from matplotlib import ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------

# Generate Data
# importing random module
import random
#random.seed(0)
random.seed(3)

A = 50 
B = -.3

print('Given parameters A = ', A, ' and B = ', B)

# Add noise to the data
L = 21
N = L
STDEV = 1.5
x = np.linspace(0, L-1, N)
x = np.array(x, dtype=np.longdouble)

y_true = A*np.exp(B*x)
y = y_true + np.random.normal(0, STDEV, N)

data = pd.DataFrame(np.column_stack((x, y)), columns= ['x','y'])
x = data['x']
y = data['y']

# Initial Scatter Plot
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.set_title(r'Scatter Plot Exercise 1')
ax.set_xlabel('Input')  # Add an x-label to the axes.
ax.set_ylabel('Output')  # Add a y-label to the axes.
ax.grid()

# Linear Regression using logarithm transformation 
# Transform y
# Vertical shift, since there are zeros in the data and ln(0) is undefined. 
C = np.abs(y.min())+1
# Logarithm Transformation
ylog = np.log(y+C)
xc = sm.add_constant(x)
lm = sm.OLS(ylog, xc)
result = lm.fit()

rp = result.params

# Construct y hat: Model Values in Exponential Form
Ah = rp[0]
Bh = rp[1]
yhat = np.exp(Ah)*np.exp(Bh*x) - C
# Convert to Pandas Series so we can computer Correlation to compute R^2
yhat = pd.Series(yhat)


# -----------------------------------------------------------------------------
# Use Scipy curve_fit

def func(x, a, b, c):
    return a * np.exp(-b * x) + c
# popt contains the parameters a, b, c and pcov is a correlation matrix
popt, pcov = curve_fit(func, x, y)
y_curvefit = func(x, *popt)
y_curvefit = pd.Series(y_curvefit)

# Contruction Plot
xx = np.linspace(x.min(),x.max(), 100)

fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(x, y, facecolor='black', ec="black", label="Data")
#ax3.plot(x, yfitexp, color='blue', label="exp(b*x) as Basis")
ax3.plot(x, y_curvefit, color='green', label='SciPy Curve Fit')
ax3.plot(xx, A*np.exp(B*xx), color='black', label='Actual', linestyle='dashed')
ax3.plot(x, yhat, label='Transform ln(y)', color='blue', linestyle='dotted')
#ax3.plot(xx, np.exp(Ah)*np.exp(Bh*xx) - C, label='Best unTransform', color='red')
ax3.set_title(r'Function vs. Model vs. Data')
ax3.set_xlabel('Input')  # Add an x-label to the axes.
ax3.set_ylabel('Output')  # Add a y-label to the axes.
ax3.grid()
ax3.legend()

print('\nExponential Model (Transorm y -> ln(y), Use Parameters) R-Squared = ', pow(yhat.corr(y),2))
#print('Exponential Model exp(b*x) as Basis Function = ', pow(yfitexp.corr(y),2))
print('Exponential Model SciPy Curve Fit = ', pow(y_curvefit.corr(y),2))
print('\n')
