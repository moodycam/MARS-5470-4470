# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%
import matplotlib
import numpy as np
import matplotlib.pyplot as plt
#%%
""" Setting Up the Plot """
#%%
fig, ax = plt.subplots(figsize=(2,2))
#%%
fig, ax = plt.subplots(figsize=(3,3), nrows=2, ncols=2)
ax[0,0].set_facecolor('black')
#%%
""" Line Plots """
#%%
t = np.linspace(-np.pi, np.pi, 64, endpoint=True)
c = np.cos(t)
s = np.sin(t)
#%%
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(t, s, label = 'sine', linewidth=2, linestyle='--', color='red')
ax.plot(t,c, label = 'cosine', linewidth=2, linestyle='-.', color='#A9A9A9')
plt.legend()
#%%
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(t, s, label = 'sine', linewidth=2, linestyle='-', color='red',
        marker='o', markersize=5)
ax.plot(t,c, label = 'cosine', linewidth=2, linestyle='-', color='#A9A9A9')
plt.legend()
#%%
fig, ax = plt.subplots(figsize=(6,3), ncols = 2)
ax[0].plot(t,s)
ax[1].plot(t,c)
#%%
fig, ax = plt.subplots(figsize=(6,3), ncols = 2, sharey=True, sharex=True)
ax[0].plot(t,s)
ax[1].plot(t,c)
#%%
fig, ax = plt.subplots(figsize=(5,5))
ax.plot(t, c)
ax.plot(t,s)
########
ax.set_title('Sines and Cosines', fontsize=18)
ax.set_xlabel('Time')
ax.set_ylabel('Magnitude')
ax.set_ylim([-2,2])
ax.set_xlim([-3,3])
#%%
ax.set('Sines and Cosines', xlabel='Time', ylabel='Magnitude', ylim=([-2,2]), xlim=([-3,3]))
#%%
plt.style.available
#%%
plt.style.use('seaborn-bright')
fig,ax = plt.subplots(figsize=(5,5))
ax.plot(t,s)
#%%
""" Meshes """
#%%
import cartopy #Couldn't get to work

import cartopy.crs as ccrs
import cartopy.feature as cfeature
#%%
fig, ax = plt.subplots(figzise=(5,5), subplot_kw=dict(projection=ccrs.Robinson()))