"""
Empirical Modeling Part 7: Curve Fitting with Power and Logistic Functions



MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay
@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
# Example 1 (Example 4.7:Cost of Advertising from text)
# Load data
file = "https://raw.githubusercontent.com/michael-b-scott/math374csumb/main/data-meeting19-book4-7.csv"
data = pd.read_csv(file)


x = data['Year']-1900 #We take 2-digit years for convenience
y = data['Expenditure (in millions)']

# Exponential Model
ylog = np.log(y) #Transform y data

# Linear Regression of Transformed Data
xc = sm.add_constant(x)
lmlog = sm.OLS(ylog, xc)
resultlog = lmlog.fit()

# Coefficients and RSquared
rplog = resultlog.params 

yexplog = np.exp(rplog[0])*np.exp(rplog[1]*x)
yexplog = pd.Series(yexplog)


# Exponential Model with SciPy Curve Fit
def func(x,a,b):
    return a * np.exp(b * x)
popt, pcov = curve_fit(func, x, y, bounds=((0,0),(np.inf,np.inf)))
y_curvefit = func(x, *popt)
y_curvefit = pd.Series(y_curvefit)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.plot(x, yexplog, color='green', label="Log Transform Method")
ax.plot(x, y_curvefit, color='blue', label="Curve Fit Model")
ax.set_title('Advertising Expenditure Exponential Model')
ax.set_xlabel('Year')  # Add an x-label to the axes.
ax.set_ylabel('Expenditures in Millions')  # Add a y-label to the axes.
ax.grid()
ax.legend()


# Residual Plot Exponential Model Curve Fit
resids = y - y_curvefit
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(x, resids, facecolor='black', ec="black", label="Data")
ax3.axhline(y=0, color='black')
ax3.set_title('Plot of Residuals Exponential Model')
ax3.set_xlabel('Year')  # Add an x-label to the axes.
ax3.set_ylabel('Residuals')  # Add a y-label to the axes.
ax3.grid()


# Power Model
# Book Method Log Transform
xlog = np.log(x)
xlogc = sm.add_constant(xlog)
lmpowlog = sm.OLS(ylog, xlogc)
resultpowlog = lmpowlog.fit()
rppowlog = resultpowlog.params
ypowlog = np.exp(rppowlog[0])*pow(x, rppowlog[1])
ypowlog = pd.Series(ypowlog)


# SciPy curve_fit Method
def power_fit(x,a,b):
    return a * x ** b
popt2, pcov2 = curve_fit(power_fit, x, y)
ypowerfit = power_fit(x, *popt2)
ypowerfit = pd.Series(ypowerfit)


fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.plot(x, ypowlog, color='green', linestyle='dashed', label="Log Transform Method")
ax.plot(x, ypowerfit, color='blue', label="curve_fit Method")
ax.set_title('Advertising Expenditure Power Model')
ax.set_xlabel('Year')  # Add an x-label to the axes.
ax.set_ylabel('Expenditures in Millions')  # Add a y-label to the axes.
ax.grid()
ax.legend()

# Sum of Squares of the Residuals Comparison between log transform method and
#   curve_fit method.

SSRLT = np.sum((y - ypowlog)**2)
print('\n SS_Res Log Transform = ', round(SSRLT))

SSRCF = np.sum((y - ypowerfit)**2)
print('\n SS_Res Curve Fit = ', round(SSRCF))

# Residual Plot Power Model Curve Fit
residspow = y - ypowerfit
fig3, ax3 = plt.subplots(figsize=(8, 6))
ax3.scatter(x, residspow, facecolor='black', ec="black", label="Data")
ax3.axhline(y=0, color='black')
ax3.set_title('Plot of Residuals Power Function Model')
ax3.set_xlabel('Year')  # Add an x-label to the axes.
ax3.set_ylabel('Residuals')  # Add a y-label to the axes.
ax3.grid()

# Power vs. Exponential Model Comparisions
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.plot(x, y_curvefit, color='green', linestyle='dashed', label="Exponential Model")
ax.plot(x, ypowerfit, color='blue', label="Power Model")
ax.set_title('Advertising Expenditure Power vs. Exponential Models')
ax.set_xlabel('Year')  # Add an x-label to the axes.
ax.set_ylabel('Expenditures in Millions')  # Add a y-label to the axes.
ax.grid()
ax.legend()

# SS_Res Comparisons
SSREXP = np.sum((y - y_curvefit)**2)
print('\n SS_Res Exponential = ', round(SSREXP))
print('\n SS_Res Power Model = ', round(SSRCF))
print('\nDifference = ', round(SSREXP-SSRCF))
print('\nRatio Exp/Pow = ', SSREXP/SSRCF)


# R Squared Comparisons
print('\nExponential Transform Log Model R-Squared = ', pow(yexplog.corr(y),2))
print('\nExponential Curve Fit Model R-Squared = ', pow(y_curvefit.corr(y),2))
print('\nPower Transform log Model R-Squared = ', pow(ypowlog.corr(y),2))
print('\nPower Curve Fit Model R-Squared = ', pow(ypowerfit.corr(y),2))