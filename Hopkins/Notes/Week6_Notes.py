# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:52:23 2019

@author: Miles
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#%%
"""
Mesh Grids
"""
#%%
""" Plots a 2D function. When we wanted to plot a 1D function we used an
np.arange or np.linspace"""
#%% np.linspace review
x = np.linspace(-np.pi*2, np.pi*2, 50) # the x values to evaluate a function at
# We can use the x values above to evaluate any function
y1 = np.sin(x)
y2 = 0.1* x**2 -1
#%%
plt.plot(x,x, '.k')
plt.plot(x, y1)
plt.plot(x, y2)
#%% np.arange review
x = np.arange(-np.pi*2, np.pi*2, 0.5) # the x values to evaluate a function at
    # the last argument in a range is the increment or spacing
        # also 'start', 'stop', and 'step'
# we can use the x values above to evaluate any function
y1 = np.sin(x)
y2 = 0.1* x**2 -1
#%%
plt.plot(x,x, '.k')
plt.plot(x, y1)
plt.plot(x, y2)
#%%
x = np.linspace(-np.pi*2, np.pi*2, 50)
y = np.linspace(-1,1,50)
X, Y = np.meshgrid(x,y)
#%%
plt.plot(x,y)
#%%
plt.plot(X,Y)
#%%
X.shape
#%%
plt.pcolormesh(X)
plt.colorbar()
#%%
Y.shape
#%%
plt.pcolormesh(Y)
plt.colorbar()
#%%
plt.pcolormesh(z)
#%%
z = np.sin(X*Y)
plt.pcolormesh(X, Y, z)
plt.colorbar()
