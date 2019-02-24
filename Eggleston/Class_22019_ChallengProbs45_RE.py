#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 20 13:41:11 2019

@author: roryeggleston
"""
import numpy as np
#%%
import xarray as xr
#%%
import cartopy as ctpy
#%%
import matplotlib.pyplot as plt
import matplotlib as mpl
#%%
data = xr.open_dataset('/Users/roryeggleston/Downloads/CESM.003.SST.1980.nc')
lat = np.array(data.lat)
lon = np.array(data.lon)
data = np.array(data.SST)
#%%
meansst = np.mean(data, axis=0)
#%%
#PROBLEM 4
print(lat)
#%%
meansst = np.mean(data, axis = 0)
meansst.shape
#%%
mask2 = np.zeros_like(meansst)
mask2.shape
#%%
masktrops = ((lat <= 23.4368) & (lat >= -23.4368))
print(masktrops)
#%%
n=0
for i in range (0,360):
    mask2[:,i] = masktrops
    n +=1
#%%
plt.pcolormesh(mask2)
#%%
masksst = np.ma.masked_where(mask2,meansst)
#%%
trop = 23.4368
#%%
fig, ax = plt.subplots(figsize=(8,3))
p = ax.pcolormesh(lon, lat, masksst, cmap='autumn_r', vmin=-2, vmax=30)
cb = plt.colorbar(p, extend='both')
cb.set_label('Sea Surface Temperature [$^{o}$C]')
ax.set(title='CESM Mean Sea Surface Temperature [$^{o}$C] Above/Below Tropics 1980', ylabel='Lat', xlabel = 'Lon (Deg E)')
#%%
#PROBLEM 5
lat
#%%
np.where(lat>26.1118)
#%%
lat[116]
#%%
np.where(meansst > 20)
#%%
lonind, latind = np.where(meansst > 20)
#%%
meansst.shape
#%%
lonind
#%%
latind
#%%
lat[21]
#%%
meansst[53,21]
#%%
np.where(lat>26.1118)
#%%
lonind = np.where(lat>26.1118)
#%%
lonind?
#%%
lonind[0]
#%%
lonind[0][0]
#%%
lat[lonind[0][0]]
#%%
mylon = 360 - 97.1681
mylon
#%%
np.where(lon > 262.8319)[0][0]
#%%
meansst.shape
#%%
meansst[116,263]
#this is on land, not in the ocean, so need to move east a bit
#%%
plt.pcolormesh(lon,lat, meansst)
plt.scatter(lon[263], lat[116])
plt.xlim(250,300)
plt.ylim(20,40)
#%%
meansst[116, 264]
#%%
plt.pcolormesh(lon,lat, meansst)
plt.scatter(lon[264], lat[116])
plt.xlim(250,300)
plt.ylim(20,40)
#%%
#MESH LECTURE
x = np.linspace(-np.pi*2, np.pi*2, 50)
y1 = np.sin(x)
y2 = 0.1* x**2 -1
plt.plot(x,x, '.k')#dot is the marker
plt.plot(x,y1)
plt.plot(x,y2)
#%%
#x = np.linspace(-np.pi*2, np.pi*2, 150) last number changes number of points
x = np.arange(-np.pi*2, np.pi*2, 0.5)

y1 = np.sin(x)
y2 = 0.1* x**2 -1

plt.plot(x,x, '.k')
plt.plot(x,y1)
plt.plot(x,y2)
#%%
x = np.linspace(-np.pi*2, np.pi*2, 50)
y = np.linspace(-1,1,50)
X,Y = np.meshgrid(x,y)
#%%
#EXERCISE What are X and Y? Plot them
X.shape
plt.pcolormesh(X)
#X is -2pi, 2pi
#%%
Y.shape
#Y is -1,1
#%%
plt.pcolormesh(Y)
#%%
z = np.sin(X*Y)
#%%
plt.pcolormesh(z)
#%%
plt.pcolormesh(x,y,z)
plt.pcolorbar
#%%
z = np.sin(X*Y)
plt.pcolormesh(X,Y,z)
plt.colorbar()
#%%
z = np.sin(x*y)
#EXERCISE could we use x and y to define z above? Why or why  not?
#You cannot define z using x and y and get the same result as in the above examples, because using only x and y will give you only one of the possible functions
#%%
#EXERCISE redo exercises 3 and 4 using meshgrids instead of for loops
#%%
#3
meansst = np.mean(data, axis = 0)
meansst.shape
#%%
LON,LAT = np.meshgrid(lon, lat)
#%%
LON
#%
LAT
#%%
plt.pcolormesh(LAT)
plt.colorbar()
#%%
maskLAT = ((LAT >= 23.4368) | (LAT <= -23.4368))
maskLAT
#%%
maskSST = np.ma.masked_where(maskLAT,meansst)
#%%
trop = 23.4368
#%%
fig, ax = plt.subplots(figsize=(8,1))
p = ax.pcolormesh(LON, LAT, maskSST, cmap='autumn_r', vmin=-2, vmax=30)
cb = plt.colorbar(p, extend='both')
cb.set_label('Sea Surface Temperature [$^{o}$C]')
ax.set(title='CESM Mean Sea Surface Temperature [$^{o}$C] in Tropics 1980', ylabel='Lat', xlabel = 'Lon (Deg E)', ylim = [-trop,trop+1])
#%%
#4
meansst = np.mean(data, axis = 0)
meansst.shape
#%%
LON,LAT = np.meshgrid(lon, lat)
#%%
LON
#%
LAT
#%%
plt.pcolormesh(LAT)
plt.colorbar()
#%%
maskLAT2 = ((LAT <= 23.4368) & (LAT >= -23.4368))
maskLAT2
#%%
maskTROPSST = np.ma.masked_where(maskLAT2,meansst)
#%%
fig, ax = plt.subplots(figsize=(8,3))
p = ax.pcolormesh(LON, LAT, maskTROPSST, cmap='autumn_r', vmin=-2, vmax=30)
cb = plt.colorbar(p, extend='both')
cb.set_label('Sea Surface Temperature [$^{o}$C]')
ax.set(title='CESM Mean Sea Surface Temperature [$^{o}$C] Above/Below Tropics 1980', ylabel='Lat', xlabel = 'Lon (Deg E)')
#%%
#EXERCISE Make a line plot of the monthly temperature at this location
plt.pcolormesh(lon,lat, meansst)
plt.scatter(lon[264], lat[116])
plt.xlim(250,300)
plt.ylim(20,40)
#%%
SPIind = np.where(lat > 26.1118)
SPItemp = SPIind[0]
#%%
SPIlat = SPItemp[0]
#%%
SPIindlon = np.where(lon > 262.8319)
SPItemp2 = SPIindlon[0]
#%%
SPIlon = SPItemp2[0]
#%%
SPIsst = data[0:11,SPIlat,SPIlon+1]
SPIsst
#%%
plt.plot(SPIsst)
#%%
#1
#It was hottest in September, at around 28.3 degrees C
#2
#It was coldest in February, at around 20.1 degrees C
#%%
#3
np.max(SPIsst)#28.368649
np.min(SPIsst)#20.171156
#%%
