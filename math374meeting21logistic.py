"""
Empirical Modeling Part 7: Curve Fitting with Logistic Functions



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
from scipy.optimize import curve_fit

# -----------------------------------------------------------------------------
# Example Logistic
# Load data
file = "https://raw.githubusercontent.com/michael-b-scott/math374csumb/main/data-meeting21-logistic.csv"
data = pd.read_csv(file)


x = data['input']
y = data['output']

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.set_title('Logisitic Growth Type Data')
ax.set_xlabel('Input')  # Add an x-label to the axes.
ax.set_ylabel('Output')  # Add a y-label to the axes.
ax.grid()
ax.legend()


# Logistic Model with SciPy Curve Fit
def logifunc(x,K,C,r):
    return K / (1 + C*np.exp(-r*(x)))

popt, pcov = curve_fit(logifunc, x, y, bounds=((0,0,-np.inf),(np.inf,np.inf,np.inf)))
y_curvefit = logifunc(x, *popt)
y_curvefit = pd.Series(y_curvefit)

fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(x, y, facecolor='black', ec="black", label="Data")
ax.plot(x, 100/(1+10*np.exp(-0.25*x)), color='black', linestyle='dashed', label="Actual")
ax.plot(x, y_curvefit, color='blue', label="Logistic Model")
ax.set_title('Logisitic Model with Data')
ax.set_xlabel('Input')  # Add an x-label to the axes.
ax.set_ylabel('Output')  # Add a y-label to the axes.
ax.grid()
ax.legend()

print('\nLogistic Curve Fit Model R-Squared =  = ', pow(y_curvefit.corr(y),2))
