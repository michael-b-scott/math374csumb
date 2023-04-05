# -*- coding: utf-8 -*-
"""
Empirical Modeling Part 5: Curve Fitting with Power and Exponential Functions


MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay
@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import matplotlib.pyplot as plt
from matplotlib import ticker
import numpy as np
import pandas as pd
import statsmodels.api as sm

# -----------------------------------------------------------------------------



# -----------------------------------------------------------------------------
# Example 1 (Example 4.7:Cost of Advertising from text)
# Load data
file = "https://raw.githubusercontent.com/michael-b-scott/math374csumb/main/data-meeting19-book4-7.csv"
data = pd.read_csv(file)

x = data['Year']
y = data['Expenditure (in millions)']

# Scatter Plot
print('\ny-data description:\n')
print(y.describe())

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.set_title('Advertising Expenditure Scatter Plot')
#ax.xaxis.set_major_locator(x(integer=True))
ax.xaxis.set_major_locator(ticker.MultipleLocator(2.00))
ax.set_xlabel('Year')  # Add an x-label to the axes.
ax.set_ylabel('Expenditures in Millions')  # Add a y-label to the axes.
ax.grid()

# Linear Regression OLS
xc = sm.add_constant(x)
lm = sm.OLS(y, xc)
result = lm.fit()
#print('\nRegression Summary:\n', result.summary())

# Coefficients and RSquared
rp = result.params
print('\nCoefficients\n')
print('Constant = {:.2f}'.format(rp[0]))
print('Year = {:.2f}'.format(rp[1]))

print('\nR2: ', result.rsquared)

# Scatter Plot with Line of Best Fit
yfit = result.fittedvalues
fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x, y, facecolor='black', ec="black", label="Data")
ax2.plot(x, yfit, color='black', label="OLS")
ax2.set_title(r'Linear Regression Model $y=\beta_0 + \beta_1 x$')
ax2.xaxis.set_major_locator(ticker.MultipleLocator(2.00))
ax2.set_xlabel('Year')  # Add an x-label to the axes.
ax2.set_ylabel('Expenditures in Millions')  # Add a y-label to the axes.
ax2.grid()

# Scatter Plot with Residuals
resids = result.resid
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(x, resids, facecolor='black', ec="black", label="Data")
ax3.axhline(y=0, color='black')
ax3.set_title('Plot of Residuals')
ax3.xaxis.set_major_locator(ticker.MultipleLocator(2.00))
ax3.set_xlabel('Year')  # Add an x-label to the axes.
ax3.set_ylabel('Expenditures in Millions')  # Add a y-label to the axes.
ax3.grid()

# -----------------------------------------------------------------------------
# Exponential Model
ylog = np.log(y) #Transform y data

# Linear Regression of Transformed Data
lmlog = sm.OLS(ylog, xc)
resultlog = lmlog.fit()
print('\nRegression Summary:\n', resultlog.summary())

# Coefficients and RSquared
rplog = resultlog.params
print('\nCoefficients Exponential Model\n')
print('Constant = {:.5f}'.format(rplog[0]))
print('Year = {:.5f}'.format(rplog[1]))

print('\nR2: ', resultlog.rsquared)

# Scatter Plot with Line of Best Fit Exponential Model
yfitlog = resultlog.fittedvalues
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.scatter(x, ylog, facecolor='black', ec="black", label="Data")
ax4.plot(x, yfitlog, color='black', label="OLS")
ax4.set_title(r'Linear Regression Exponential Model')
ax4.xaxis.set_major_locator(ticker.MultipleLocator(2.00))
ax4.set_xlabel('Year')  # Add an x-label to the axes.
ax4.set_ylabel('Expenditures in Millions')  # Add a y-label to the axes.
ax4.grid()

# Scatter Plot with Exponential Curve
yfitexp = np.exp(rplog[0])*np.exp(rplog[1]*x)

print('Constant = {:.5f}'.format(np.exp(rplog[0])))

fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.scatter(x, y, facecolor='black', ec="black", label="Data")
ax5.plot(x, yfitexp, color='black', label="OLS")
ax5.set_title(r'Linear Regression Exponential Model')
ax5.xaxis.set_major_locator(ticker.MultipleLocator(2.00))
ax5.set_xlabel('Year')  # Add an x-label to the axes.
ax5.set_ylabel('Expenditures in Millions')  # Add a y-label to the axes.
ax5.grid()

print('\nExponential Model R-Squared =  = ', pow(yfitexp.corr(y),2))
# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------


# -----------------------------------------------------------------------------
