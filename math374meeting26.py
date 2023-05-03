"""
Continuous Models Part 4 - The Classical Predator-Prey Model



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

# ----------------------------------------------------------------------------
# 2x2 Systems

# Define Function for System
a = 0.4
b = 0.01
c = 0.003
d = 0.3
def de_system(t,ys):
    x = ys[0]
    y = ys[1]
    return np.array([x*(a-b*y), y*(c*x-d)])

# Critical Points
cp = pd.DataFrame([[0,0], [d/c, a/b]], columns = ['x', 'y']) #Empty dataframe

# Generate Vector Field
x = np.linspace(0, 300, 30)
y = np.linspace(0, 100, 20)
X, Y = np.meshgrid(x, y)

dx, dy = de_system(0, [X, Y])

# Normalize Arrow Length
dx = dx / np.sqrt(dx**2 + dy**2);
dy = dy / np.sqrt(dx**2 + dy**2);

# Plot vector field with 5 solution curves
fig, ax = plt.subplots(figsize=(15, 15))
ax.scatter(cp['x'], cp['y'], s=80, facecolor='black', ec="black", label="Critical Points")
ax.quiver(X, Y, dx, dy, color='k', width = 0.003)
#ax.streamplot(X, Y, dx, dy, density=.8, linewidth = 1, broken_streamlines=False)
for i in range(5):
    k = 40*2*i/(2*i+1) + 12/(i+1)
    y0 = (i/2+1)*12
    #print('y0 = ', y0)
    initial_conditions = np.array([100, y0])
    sol = integrate.solve_ivp(de_system, (0, 20),
       initial_conditions, max_step=0.01)
    ax.plot(sol.y[0, :], sol.y[1, :], color='b', linewidth=5)
ax.grid()
ax.set_title("Predator Prey System", fontsize="25")
ax.set_xlabel("Prey", fontsize="20")
ax.set_ylabel("Predator", fontsize="20")
ax.tick_params(axis='both', which='major', labelsize=20)

# ----------------------------------------------------------------------------
# Plot Predator and Prey Populations vs. time

# Initial Conditions
x0 = 50
y0 = 20
initial_conditions = np.array([x0, y0])
sol = integrate.solve_ivp(de_system, (0, 60),
   initial_conditions, max_step=0.01)

fig, ax = plt.subplots(figsize=(15, 15))
ax.plot(sol.t, sol.y[0, :], color='b', linewidth=5, label='Prey')
ax.plot(sol.t, sol.y[1, :], 'g--', label='Predator', linewidth=5)
ax.grid()
ax.set_title("Predator Prey System", fontsize="25")
ax.set_xlabel("Time", fontsize="20")
ax.set_ylabel("Populations", fontsize="20")
ax.tick_params(axis='both', which='major', labelsize=20)
ax.legend(fontsize="20", loc ="upper right")