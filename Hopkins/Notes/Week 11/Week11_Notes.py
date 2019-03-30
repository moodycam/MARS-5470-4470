# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:00:02 2019

@author: Miles
"""

#%%
"""
dn/dt = aN
N(t) = C * e^at
    C = constant
"""
#%%
import numpy
import matplotlib.pyplot as plt
#%%
""" Numerical Solutions Using Python """
"""
A simple python code for solving these equations is shown below.
"""
#%%
# set the initial parameters
# alpha = 1.
# beta = 1.2
# gamma = 4.
# delta = 1.
alpha = 1. # Pop. growth of prey
beta = 1.2 # Rate at which predator and prey meet
gamma = 4. # Death rate of predators
delta = 1. # Rate of change
#%%
#define the time stepping scheme - euler forward, as used in earlier lessons
def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    
    return u + dt * f(u)
#%%
# define the function that represents the Lotka-Volterra equations
def f(u):
    """Returns the rate of change of species numbers.
    
    Parameters
    ----------
    u : array of float
        array containing the solution at time n.
        
    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """
    x = u[0]
    y = u[1]
    return numpy.array([x*(alpha - beta*y), -y*(gamma - delta*x)])
#%%
# set time-increment and discretize the time
T  = 15.0                           # final time
dt = 0.01                           # set time-increment
N  = int(T/dt) + 1                  # number of time-steps
x0 = 10.
y0 = 2.
t0 = 0.

# set initial conditions
u_euler = numpy.empty((N, 2))

# initialize the array containing the solution for each time-step
u_euler[0] = numpy.array([x0, y0])

# use a for loop to call the function rk2_step()
for n in range(N-1):
    
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = numpy.linspace(0.0, T,N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_euler, label = 'prey ')
plt.plot(time, y_euler, label = 'predator')
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("predator prey model")
#%%
""" System Behavior """
"""
A better understanding of the system behaviour can be obtained by a phase plot
 of the population of predators vs. the population of prey. It will tell us if
 the system sustains or collapses over time. For the choice of parameters
 $ \alpha, \beta, \gamma $ and $ \delta $ made above, we see that the maximum
 population of each species keeps increasing each cycle. You can read more
 about that in the Wikipedia link mentioned above.
"""
#%%
plt.plot(x_euler, y_euler, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
#labels
plt.xlabel("number of prey")
plt.ylabel("number of predators")
#title
plt.title("predator prey model")
#%%
""" 10.2 """
#%%
import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import geopandas as gpd
from descartes import PolygonPatch
#%%
def load_shape_file(filepath):
    """Loads the shape file desired to mask a grid.
    Args:
        filepath: Path to *.shp file
    """
    shpfile = gpd.read_file(filepath)
    return shpfile
#%%
shp = load_shape_file('C:\\Users\Miles\Downloads\LME66\LMEs66.shp')
shp.head() # .head gives you the first few lines, aka shapes
#%%
shp
#%%
print(shp)
#%%
from mpl_toolkits.basemap import Basemap
#%%
# Basic plot in basemap

ax = plt.figure(figsize=(16,20), facecolor = 'w')

# limits of plot
limN, limS, limE, limW = 84.,-80.,180,-180

#m = Basemap(projection='hammer',lon_0=0)
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#BDA973', lake_color='#BDA973');
