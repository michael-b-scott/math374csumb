"""
Continuous Models Part 3 - 
    Systems of First-Order Homogeneous Linear Differential Equations With
	Constant Coefficients



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
from scipy import integrate
from scipy.optimize import fsolve
from numpy.linalg import eig


# 2x2 Systems with Constant Coefficients

A = np.array([[3.0, 5.0],
              [3.0, 1.0]])


w,v = eig(A)
# Eigenvectors
# This gives normalized (unit “length”) eigenvectors, such that the column
#   v[:,i] is the eigenvector corresponding to the eigenvalue w[i].
print('\nE-values:', w)
print('E-vectors:\n', v, '\n')

def de_system(t, ys):
    x = ys[0]
    y = ys[1]
    return np.array([A[0,0]*x+A[0,1]*y, A[1,0]*x+A[1,1]*y])


# Generate Vector Field
x = np.linspace(-5.0, 5.0, 20)
y = np.linspace(-5.0, 5.0, 20)
X, Y = np.meshgrid(x, y)

#dx, dy = de_system(0, [X, Y])

dx = A[0,0]*X+A[0,1]*Y
dy = A[1,0]*X+A[1,1]*Y

# Normalize Arrow Length for Vector Field Using Quiver Function
N = np.sqrt(pow(dx,2)+pow(dy,2))
dx /= N
dy /= N


# Phase Diagram with Solution Plot
fig, ax = plt.subplots(figsize=(15, 15))

# Vector Field
ax.quiver(X, Y, dx, dy, color='k', width = 0.003)
# Solution Curves
ax.streamplot(X, Y, dx, dy, density=.8, linewidth = 1, broken_streamlines=True)
# Equilibrium or Fixed Point at (0,0)
ax.scatter(0,0)
# Plot Eigenvectors (Equations of lines)
ax.plot(x, v[1,0]*x/v[0,0], linestyle='dashed', color='k', linewidth=1.0, label=r'$v_1$')
ax.plot(x, v[1,1]*x/v[0,1], linestyle='dashdot', color='k', linewidth=1.0, label=r'$v_2$')

# Plot parameters
ax.set_xlim(-5, 5)
ax.set_ylim(-5, 5)
ax.set_title("2 X 2 System", fontsize="20")
ax.set_xlabel("x", fontsize="20")
ax.set_ylabel("y", fontsize="20")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.grid()
ax.legend(fontsize="20", loc ="upper right");
