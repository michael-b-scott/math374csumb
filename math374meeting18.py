# -*- coding: utf-8 -*-
"""
Empirical Modeling Part 4: Periodic Data

MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay
@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt


# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# Linear Regression with periodic data
file = "https://raw.githubusercontent.com/michael-b-scott/math374csumb/main/data-meeting18-periodic.csv"
data = pd.read_csv(file)

x = data['x']
y = data['y']

print('\ny-data description:\n')
print(y.describe())

fig2, ax2 = plt.subplots(figsize=(8, 6))
ax2.scatter(x, y, facecolor='black', ec="black", label="Data")
#ax2.plot(x, yfit, color='black', label="OLS")
#ax2.plot(dh[49] ,result.fittedvalues[49], marker="o", markeredgecolor="black", markerfacecolor="black")
ax2.set_title('Linear Regression Periodic Scatter Plot')
ax2.set_xlabel('x')  # Add an x-label to the axes.
ax2.set_ylabel('y')  # Add a y-label to the axes.
ax2.grid()

# Linear Regression First Try
xc = sm.add_constant(x)
lm = sm.OLS(y, xc)
result = lm.fit()
print('\nFirst Try Regression Summary:\n', result.summary())

#Coefficients
#rp = result.params
#print('\n Coefficients \n', rp)

yfit = result.fittedvalues
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(x, y, facecolor='black', ec="black", label="Data")
ax3.plot(x, yfit, color='black', label="OLS")
#ax2.plot(dh[49] ,result.fittedvalues[49], marker="o", markeredgecolor="black", markerfacecolor="black")
ax3.set_title(r'Linear Regression Periodic First Try $y=\beta_0 + \beta_1 x$')
ax3.set_xlabel('x')  # Add an x-label to the axes.
ax3.set_ylabel('y')  # Add a y-label to the axes.
ax3.grid()

# Linear Regression Second Try
f1 = np.sin(2*np.pi*x)
f2 = np.cos(2*np.pi*x)
X = np.column_stack((f1, f2, np.ones(len(x))))
lm = sm.OLS(y, X)
result = lm.fit()
print('\nSecond Try Regression Summary:\n', result.summary())





yfit = result.fittedvalues
rp = result.params
xx = np.linspace(0, 10, 10)
yfits = rp[0]*np.sin(2*np.pi*xx) + rp[1]*np.cos(2*np.pi*xx) + rp[2]
fig4, ax4 = plt.subplots(figsize=(8, 6))
ax4.scatter(x, y, facecolor='black', ec="black", label="Data")
ax4.plot(x, yfit, color='black', label="OLS")
ax4.plot(xx, yfits, color='blue', label="OLSs")
#ax2.plot(dh[49] ,result.fittedvalues[49], marker="o", markeredgecolor="black", markerfacecolor="black")
ax4.set_title(r'Linear Regression Periodic Second Try $y=\beta_0 + \beta_1 \sin(2\pi x) + \beta_2 \cos(2\pi x)$')
ax4.set_xlabel('x')  # Add an x-label to the axes.
ax4.set_ylabel('y')  # Add a y-label to the axes.
ax4.grid()



# Linear Regression Third Try
X = np.column_stack((np.sin(0.5*np.pi*x), np.cos(0.5*np.pi*x), np.ones(len(x))))
lm = sm.OLS(y, X)
result = lm.fit()
print('\nThird Try Regression Summary:\n', result.summary())

#Coefficients
rp = result.params
#print('\n Coefficients \n', rp)

yfit = result.fittedvalues
fig5, ax5 = plt.subplots(figsize=(8, 6))
ax5.scatter(x, y, facecolor='black', ec="black", label="Data")
xx = np.linspace(0, 10, 100)
y_true = 2.5*np.sin(1.5*(xx-2.0))+6
yfits = rp[0]*np.sin(0.5*np.pi*xx) + rp[1]*np.cos(0.5*np.pi*xx) + rp[2]
ax5.plot(xx, yfits, color='black', label="OLS")
ax5.plot(xx, y_true, '--', color='blue', label="Actual")
#ax2.plot(dh[49] ,result.fittedvalues[49], marker="o", markeredgecolor="black", markerfacecolor="black")
ax5.set_title(r'Linear Regression Periodic Third Try $y=\beta_0 + \beta_1 \sin(\pi/2 x) + \beta_2 \cos(\pi/2 x)$')
ax5.set_xlabel('x')  # Add an x-label to the axes.
ax5.set_ylabel('y')  # Add a y-label to the axes.
ax5.grid()


# -----------------------------------------------------------------------------
