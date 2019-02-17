# -*- coding: utf-8 -*-
"""
Created on Mon Feb 11 13:53:13 2019

@author: Miles
"""
#%%
""" Warm-Up Challenge Problem """
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#%%
data = xr.open_dataset('C:\\Users\Miles\Downloads\CESM.003.SST.1980.nc')
data
#%%
lat = np.array(data.lat)
lon = np.array(data.lon)
time = np.array(data.time)
sst = np.array(data.SST)
#%%
plt.pcolormesh(sst[0,:,:])
#%% Plot
fig, ax = plt.subplots(figsize=(17,10), nrows=3, ncols=4, sharex=True, sharey=True)
im = ax[0, 0].pcolormesh(lon, lat, sst[0], cmap="plasma", vmin=-2, vmax=30)
ax[0, 1].pcolormesh(lon, lat, sst[1], cmap="plasma", vmin=-2, vmax=30)
ax[0, 2].pcolormesh(lon, lat, sst[2], cmap="plasma", vmin=-2, vmax=30)
ax[0, 3].pcolormesh(lon, lat, sst[3], cmap="plasma", vmin=-2, vmax=30)
ax[1, 0].pcolormesh(lon, lat, sst[4], cmap="plasma", vmin=-2, vmax=30)
ax[1, 1].pcolormesh(lon, lat, sst[5], cmap="plasma", vmin=-2, vmax=30)
ax[1, 2].pcolormesh(lon, lat, sst[6], cmap="plasma", vmin=-2, vmax=30)
ax[1, 3].pcolormesh(lon, lat, sst[7], cmap="plasma", vmin=-2, vmax=30)
ax[2, 0].pcolormesh(lon, lat, sst[8], cmap="plasma", vmin=-2, vmax=30)
ax[2, 1].pcolormesh(lon, lat, sst[9], cmap="plasma", vmin=-2, vmax=30)
ax[2, 2].pcolormesh(lon, lat, sst[10], cmap="plasma", vmin=-2, vmax=30)
ax[2, 3].pcolormesh(lon, lat, sst[11], cmap="plasma", vmin=-2, vmax=30)
# Get Individual Titles
ax[0,0].set_title('January', fontsize = (16))
ax[0,1].set_title('February', fontsize = (16))
ax[0,2].set_title('March', fontsize = (16))
ax[0,3].set_title('April', fontsize = (16))
ax[1,0].set_title('May', fontsize = (16))
ax[1,1].set_title('June', fontsize = (16))
ax[1,2].set_title('July', fontsize = (16))
ax[1,3].set_title('August', fontsize = (16))
ax[2,0].set_title('September', fontsize = (16))
ax[2,1].set_title('October', fontsize = (16))
ax[2,2].set_title('November', fontsize = (16))
ax[2,3].set_title('December', fontsize = (16))
# Colorbar
fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05, fraction=0.05, extend="both")
fig.suptitle('Sea Surface Temperatures', fontsize=24)
#%%
""" Jupyter Lecture Challenge """
#%%
data = np.loadtxt('C:\\Users\Miles\Downloads\populations.txt')
#%%
data
data.T
year, hares, lynxes, carrots = data.T
#%%
year
#%%
hares
#%%
lynxes
#%%
print(hares.mean())
print(lynxes.mean())
print(carrots.mean())
#%%
plt.plot(data[:,0], data[:,1])
plt.plot(data[:,0], data[:,2])
plt.plot(data[:,0], data[:,3])
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(['Hares', 'Lynxes', 'Carrots'], loc = 'upper right')
plt.title('Populations of Hares, Lynxes, and Carros from 1900 to 1920', fontsize=16)
#%%
""" Challenge Problem 1 """
#%%
mask = (hares >= 60000)
#%%
hares_masked = np.ma.masked_where(mask, hares)
#%%
hares_masked
#%%
# note this is the same as above
plt.plot(year, hares_masked)
plt.plot(year, lynxes)
plt.plot(year, carrots)
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(['Hares', 'Lynxes', 'Carrots'], loc = 'upper right')
#%%
print(hares_masked.mean())
print(lynxes.mean())
print(carrots.mean())
#%%
""" Challenge Problem 2 """
#%%
plt.scatter(hares,lynxes)
#%%
r = np.corrcoef(hares,lynxes)
    # r value is 0.07189206
        # very close to 0, so not very correlated
#%%
plt.scatter(hares,lynxes)
plt.title(("r=" + str(r)))
