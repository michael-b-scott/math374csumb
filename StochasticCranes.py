# -*- coding: utf-8 -*-
"""
Stochastic Model of Sandhill Crane Populuation
MATH 374: Mathematical Modeling, CSUMB

@author: Michael B. Scott
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

# -------------------------------------------------------------------------
# Generate Simulated Data
# -------------------------------------------------------------------------
years = 15
year = pd.Series(range(0, years+1))

# Initial Population
P0 = 100

# Birth and Death Rate Assumptions
# Normal
birthrate = 0.50
deathrate = 0.10

# Get simulated population data ------------------------------------------

simnum = 50 #Number of Simulations
simdata = pd.DataFrame() #Empty dataframe


for j in range(simnum):
    # Environmental Stochasticity
    # Generate random number in (0,1) from uniform distribution for each year.
    EnvRand = np.random.uniform(0, 1, years+1)
    
    #Demographic Stochasticity
    # Birth rate = 0.5 with SD = 0.03
    DemBirth = np.round(np.random.normal(birthrate, 0.03, years+1), 4)
    # Death rate = 0.1 with SD = 0.08
    DemDeath = np.round(np.random.normal(deathrate, 0.08, years+1), 4)
    DemDeath[DemDeath < 0] = 0; #Changes any negative values to zero
    
    # Generate Population
    ColName = "PopSim" + str(j+1)
    PopSto = pd.DataFrame({ColName: pd.Series(dtype='float')})
    BirthRate = pd.DataFrame({'Birthrate': pd.Series(dtype='float')})
    DeathRate = pd.DataFrame({'Deathrate': pd.Series(dtype='float')})
    
    
    for i in range(years+1):
        # Get population from previous year
        if (i == 0):
            Popm1 = P0
        else:
            Popm1 = PopSto[ColName][i-1]

   
       # print(f"Popm1={Popm1}") # Uncomment for debugging
        # Get Birthrate and Deathrate for each year
        if (EnvRand[i] < 0.04):
            BirthRate.loc[len(BirthRate.index)] = 0.6*DemBirth[i]
            DeathRate.loc[len(DeathRate.index)] = 1.26*DemDeath[i]
            #BirthRate.append(0.6*DemBirth[i])
            #DeathRate.append(1.25*DemDeath[i])
        else:
            BirthRate.loc[len(BirthRate.index)] = DemBirth[i]
            DeathRate.loc[len(DeathRate.index)] = DemDeath[i]
      
        if (i == 0):
            PopSto.loc[len(PopSto.index)] = Popm1
        else:
            PopSto.loc[len(PopSto.index)] = Popm1*(1 + DemBirth[i] - DemDeath[i])
   
    simdata = pd.concat([simdata, PopSto], axis=1)
    
# -------------------------------------------------------------------------
# Summary Statistics and Histogram for Given Year
# -------------------------------------------------------------------------

yeardata = 10;
# Summary statistics for a particular year
print('Summary Statistics for Year ',yeardata)
print(simdata.iloc[yeardata].describe())

# Histogram over all simulations for a particular year.
n_bins = 7
# Creating histogram
fig, axs = plt.subplots(1, 1, figsize =(10, 7), tight_layout = True)

axs.hist(simdata.iloc[yeardata], bins = n_bins, histtype='bar', ec='black')
axs.set_xlabel('Histogram for Year %i' %yeardata)  # Add an x-label to the axes.
# Show plot
plt.show()

# -------------------------------------------------------------------------
# Simulate population using mean value by year over all simulations -------
# -------------------------------------------------------------------------
# Get mean values of population for each year
popsim = pd.Series(dtype="float64")

for i in range(years+1):
    #popsim.append(simdata.iloc[i])
    popsim.loc[len(popsim.index)] = np.mean(simdata.iloc[i])
    
# Generate plot of simulated data (mean over all simuluations per year)
# This will be our determistic model
xx = np.linspace(0, years, 100)
# Deterministic model assuming 40% growth rate
yy = P0*1.4**(xx)

yyn = P0*1.4**(simdata.index)

fig2, ax2 = plt.subplots(figsize=(8, 4), layout='constrained')
ax2.scatter(x=range(years+1), y=popsim, s=70, facecolor='none', label=r'Population Data', ec="blue")
ax2.plot(xx, yy, label='Deterministic Model', c="darkblue")

ax2.set_title('Simulated Data with Deterministic Model')
ax2.set_xlabel('Years', fontsize=11)  # Add an x-label to the axes.
ax2.set_ylabel('Crane Population', fontsize=11)  # Add a y-label to the axes.

#ax2.legend();  # Add a legend.
ax2.grid(True)


# -------------------------------------------------------------------------
# Deterministic model with boxplots for each year and observed data -------
# -------------------------------------------------------------------------
fig3, ax3 = plt.subplots(figsize=(8, 4), layout='constrained')
ax3.plot(xx+1, yy, label='Deterministic Model', c="darkblue")

# Need to transpose dataframe to work with boxplot. 
simdata_transpose = simdata.T

# Syntax of boxplot() using Pandas boxplot. 
# Used Pandas due to issue with matplotlib boxplot offset x values for unknown reason.
simdata_transpose.boxplot(ax=ax3)


ax3.set_title('Side-by-Side Boxplots and Deterministic Model')
ax3.set_xlabel('Years', fontsize=11)  # Add an x-label to the axes.
ax3.set_ylabel('Crane Population', fontsize=11)  # Add a y-label to the axes.

#ax3.legend();  # Add a legend.
ax3.grid(True)

# -------------------------------------------------------------------------
# Line graphs of simulations
# -------------------------------------------------------------------------
fig4, ax4 = plt.subplots(figsize=(8, 4), layout='constrained')

for i in range(simnum):
    ax4.plot(range(years+1),simdata.iloc[:,i] )
    
ax4.plot(xx, yy, label='Deterministic Values', linestyle="dotted", color='darkblue')
ax4.plot(range(years+1), popsim, label='Mean Values', linestyle="dashed", color='black')

ax4.set_title('Line Graphs for %i Simulations' %simnum)
ax4.set_xlabel('Years', fontsize=11)  # Add an x-label to the axes.
ax4.set_ylabel('Crane Population', fontsize=11)  # Add a y-label to the axes.
ax4.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}'))

ax4.legend();  # Add a legend.
ax4.grid(True)
