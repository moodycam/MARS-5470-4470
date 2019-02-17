#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 13 13:43:09 2019

@author: roryeggleston
"""

import xarray as xr
import numpy as np
import matplotlib.pyplot as plt
data = xr.open_dataset('/Users/roryeggleston/Downloads/CESM.003.SST.1980.nc')
lat = np.array(data.lat)
lon = np.array(data.lon)
data = np.array(data.SST)
data
#%%
data.shape
#%%
SST = plt.figure(figsize=(8,6))
plt.subplots()
#%%
fig, ax = plt.subplots(figsize=(8,6), ncols=4, nrows=3, sharex=True, sharey=True)
ax[0,0].pcolormesh(lon, lat, data[0,:,:])
ax[0,0].set_title('January', fontsize=12)
ax[0,1].pcolormesh(lon, lat, data[1,:,:])
ax[0,1].set_title('February', fontsize=12)
ax[0,2].pcolormesh(lon, lat, data[2,:,:])
ax[0,2].set_title('March', fontsize=12)
ax[0,3].pcolormesh(lon, lat, data[3,:,:])
ax[0,3].set_title('April', fontsize=12)
ax[1,0].pcolormesh(lon, lat, data[4,:,:])
ax[1,0].set_title('May', fontsize=12)
ax[1,1].pcolormesh(lon, lat, data[5,:,:])
ax[1,1].set_title('June', fontsize=12)
ax[1,2].pcolormesh(lon, lat, data[6,:,:])
ax[1,2].set_title('July', fontsize=12)
ax[1,3].pcolormesh(lon, lat, data[7,:,:])
ax[1,3].set_title('August', fontsize=12)
ax[2,0].pcolormesh(lon, lat, data[8,:,:])
ax[2,0].set_title('September', fontsize=12)
ax[2,1].pcolormesh(lon, lat, data[9,:,:])
ax[2,1].set_title('October', fontsize=12)
ax[2,2].pcolormesh(lon, lat, data[10,:,:])
ax[2,2].set_title('November', fontsize=12)
ax[2,3].pcolormesh(lon, lat, data[11,:,:])
ax[2,3].set_title('December', fontsize=12)
im = ax[0,0].pcolormesh(lon, lat, data[0,:,:])
fig.colorbar(im, ax = ax, extend="both")
#%%
fig, ax = plt.subplots(figsize=(10,6), ncols=4, nrows=3, sharex=True, sharey=True)
cmax = 27
cmin = 0
n = 0
months = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')
for i in range(0,3):
    for j in range(0,4):
        ax[i,j].pcolormesh(lon,lat,data[n,:,:], vmin = 0, vmax =  27)
        ax[i,j].set_title(months[n])
        n +=1 #this is the counter which keeps the order correct and makes it start at0 and go to 11
im = ax[0,0].pcolormesh(lon, lat, data[0,:,:])
fig.colorbar(im, ax = ax, extend="both")