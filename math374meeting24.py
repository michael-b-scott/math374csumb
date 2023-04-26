"""
Vector Fields with Solutions Curves in Python



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


# Solving ODE of form dy/dt = f(t,y)

def f(t,y):
    return 0.25*(y**2 + y - 2)

t_range = (-4, 4)


t_vals = pd.DataFrame() #Empty dataframe
y_sols = pd.DataFrame() #Empty dataframe

t_iter = 2*(t_range[1] - t_range[0])
for i in range(0, t_iter):
    t0n = t_range[0] + i/2
    t0 = np.array([t0n])
    sol = integrate.solve_ivp(f, t_range, t0, max_step=0.1)
    t_val = pd.Series(sol.t)
    t_vals = pd.concat([t_vals, t_val], axis=1)
    y_sol = pd.Series(sol.y[0, :])
    y_sols = pd.concat([y_sols, y_sol], axis=1)



# -----------------------------------------------------------------------------
# Generate direction field

# Meshgrid
t, y = np.meshgrid(np.linspace(-4, 4, 20), 
                   np.linspace(-4, 4, 20))

# Directional vectors
u = t/t
v = f(t,y)

# Normalize Arrow Length
u = u / np.sqrt(u**2 + v**2);
v = v / np.sqrt(u**2 + v**2);
  
# Plotting Vector Field with QUIVER

fig, ax = plt.subplots(figsize=(8, 6))
ax.quiver(t, y, u, v, color='k', width = 0.003, scale_units='x', scale=3)
ax.plot(t_vals.iloc[:,0], y_sols.iloc[:,0])
for i in range(len(y_sols.axes[1])):
    ax.plot(t_vals.iloc[:,i], y_sols.iloc[:,i])
ax.set_title('Vector Field')
ax.set_xlabel('t')  # Add an x-label to the axes.
ax.set_ylabel('y')  # Add a y-label to the axes.
ax.set_ylim(-4, 4)
ax.set_ylim(-4, 4)
ax.grid()
#ax.legend()
ax.set_aspect('equal')
