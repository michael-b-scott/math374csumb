# -*- coding: utf-8 -*-
"""
Empirical Modeling Part 1
Linearity, Covariance, Linear Indedepence vs. Dependence


MATH 374 Mathematical Modeling Spring 2023
Department of Mathematics and Statistics
California State University, Monterey Bay
@author: Michael B. Scott
@email: mscott@csumb.edu
"""

# Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


dataurl = "https://raw.githubusercontent.com/michael-b-scott/math374csumb/main/data_meeting15.csv"
data = pd.read_csv(dataurl)

dataxy1 = pd.DataFrame().assign(x=data['x'], y1=data['y1'])

print('\nCovariance with Pandas\n')
print(dataxy1.cov())

print('\nCovariance with Numpy\n')
covMatrix = np.cov(dataxy1.T, bias=False)
print(covMatrix)

fig, ax = plt.subplots()
ax.scatter(x=data['x'], y=data['y1'], s=30, facecolor='none', ec="black")
ax.set_xlabel('x')  # Add an x-label to the axes.
ax.set_ylabel('y1')  # Add a y-label to the axes.