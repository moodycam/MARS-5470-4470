# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 13:53:20 2019

@author: Miles
"""
#%% Imports Packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#%% Imports Dataset
data = xr.open_dataset('C:\\Users\Miles\Downloads\CESM.003.SST.1980.nc')
data
#%% Set Working Data
lat = np.array(data.lat)
lon = np.array(data.lon)
time = np.array(data.time)
# Monthly sea surface temperature
sst = np.array(data.SST)
# Mean of SST
meand = np.mean(sst, axis=0)
#%%
""" Problem 1 """
"""
Take the annual mean and plot it with the correct lat and lon. Use a different
 colormap than the default. Add axes labels, a colorbar and title. In the
 title, include what the plotted variable is and the units.
"""
#%%
im = plt.pcolormesh(lon, lat, meand, cmap="plasma")
plt.suptitle('Annual Mean Seas Surface Temprature, 1980', fontsize=20)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.colorbar(im, orientation='vertical', pad=0.05, fraction=0.05, extend="both")
#%%
""" Problem 2 """
"""
Take seasonal averages: DJF, etc. Plot the four seasons in a 2x2 plot of
 subplots, label each plot and put on one colorbar for all the plots, ends
 extended, with a min value of 0 and a max of 30. Make a descriptive
 supertitle (suptitle) for all the plots. Would it be correct to label these
 plots Winter, etc?
"""
#%% Separate by Season
win = sst[(0,1,11),:,:] # specifies which months (in []), then all rows and
    # columns within that month
spr = sst[(2,3,4), :, :]
summ = sst[(5,6,7), :, :]
fall = sst[(8,9,10), :, :]
#%% Seasonal Means
win_mean = np.mean(sst[(0,1,11), :, :], axis=0)
spr_mean = np.mean(sst[(2,3,4), :, :], axis=0)
summ_mean = np.mean(sst[(5,6,7), :, :], axis=0)
fall_mean = np.mean(sst[(8,9,10), :, :], axis=0)
#%% Plotting
fig, ax = plt.subplots(figsize=(17,10), nrows=2, ncols=2, sharex=True, sharey=True)
im = ax[0,0].pcolormesh(lon, lat, win_mean, cmap='plasma', vmin=0, vmax=30)
ax[0,1].pcolormesh(lon, lat, spr_mean, cmap='plasma', vmin=0, vmax=30)
ax[1,0].pcolormesh(lon, lat, summ_mean, cmap='plasma', vmin=0, vmax=30)
ax[1,1].pcolormesh(lon, lat, fall_mean, cmap='plasma', vmin=0, vmax=30)
# Setting Titles
ax[0,0].set_title('Winter (DJF)', fontsize=16)
ax[0,1].set_title('Spring (MAM)', fontsize=16)
ax[1,0].set_title('Summer (JJA)', fontsize=16)
ax[1,1].set_title('Fall (SON)', fontsize=16)
fig.suptitle('Seasonal Mean Sea Surface Temperatures, 1980', fontsize=32)
# Colorbar
fig.colorbar(im, ax=ax, orientation='vertical', pad=0.05, fraction=0.05, extend="both")
#%%
""" Problem 3 """
"""
Mask out regions outside of the tropics (google definition of tropics) and
 plot the annual mean again. Adjust the font size of your axes labels and
 title. Adjust the axis limits to be constrained to the data
 (i.e. no whitespace).
"""
#%% Masking Above the Tropics
mask1 = (lat >= 23.4368) # Sets all values above this one
#%% Masking Below the Tropics
mask2 = (lat <= -23.4368) # Sets all values below this one
#%%
mask3 = np.zeros_like(meand) # Creates an array shaped exactly the same as
    # 'meand' and fills it just with 0s
#%%
""" Note: to mask something the mask must be the same dimensions as the array
    you wish to mask - otherwise it will not work """
#%%
mask3.shape
#%%
for i in range(360):
    mask3[:, i] = mask1 | mask2 # Changes all of the values in mask3 to fit
        # with the masking of mask1 and mask2
    # 'i' is the 'i-th' column, so all of the dimensions (rows) in each
        # column
    # this '|' symbol (shift \) is the symbol for 'and'    
plt.pcolormesh(mask3) # allows us to see what mask3 now loks like
#%%
trop = np.ma.masked_where(mask3, meand) # masks mask3 over meand
#%%
plt.figure(figsize=(15,3), dpi=70) # Sets Figure Size
im2 = plt.pcolormesh(lon, lat, trop, cmap='plasma', vmin=0, vmax=30) # Plots and Sets Colors
plt.ylim(-22, 23.4368) # Corrects the 'y' limits to remove white space
plt.suptitle('Annual Mean Sea Surface Temprature for the Tropics, 1980', fontsize=20)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.colorbar(im2, orientation='vertical', pad=0.05, fraction=0.05, extend="both")
#%%
""" Problem 4 """
"""
Mask out the tropics and plot again.
"""
#%%
mask4 = (lat <= 23.4368) & (lat >= -23.4368) # Just the opposite of mask 3
#%%
for i in range(360):
    mask3[:, i] = mask4

plt.pcolormesh(mask3)
#%%
good_parts = np.ma.masked_where(mask3, meand)
#%%
im2 = plt.pcolormesh(lon, lat, good_parts, cmap='plasma', vmin=0, vmax=30) # Plots and Sets Colors
plt.suptitle('Annual Mean Sea Surface Temprature for the Areas of the Earth With Liveable Fucking Temperatures, 1980', fontsize=20)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.colorbar(im2, orientation='vertical', pad=0.05, fraction=0.05, extend="both")
#%%
""" Problem 5 """
"""
(Advanced) Find the grid cell closest to us. How big is this cell in square km
 and miles? Plot the seasonal cycle of SST. What are our local seasonal
 averages, according to this model? Annual Mean? Are they realistic? What data
 sources could you use to validate this?
"""
#%%
""" Provided in the Lecture Notes """
#%%
# Latitutde of South Padre Island is 26.1118 N, 97.1681 W
lat
#%% use np.where to find elements that are greater than a given value
np.where(lat>26.1118)
#%%
lat[116]
#%%
""" Aside, finding values in a 2D array """
#%% if we had a 2D array it would give us two arrays, one for each dimension
np.where(meand > 20)
#%% we can assign arrays to these indicies where the constraint is true
    # where is temp greater than 20
lon_ind, lat_ind = np.where(meand > 20)
#%% repeats numbers because it provides indexes
lon_ind
#%%
lon[53]
#%%
lat_ind
#%%
lat[21]
#%% yes, greater than 20
meand[53,21]
#%%
""" Back to the Main Problem """
#%%
np.where(lat>26.1118)
#%% Creates an array of points that are indexing all of the lat and lon at a 
    # SST that is higher than 20 C
lonind = np.where(lat>26.1118)
#%%
lonind?
#%% Accessess the first element of the tupel
lonind[0]
#%%
lonind[0][0]
#%%
lat[lonind[0][0]]
#%% note lon is 0 to 360 East
mylon = 360 - 97.1681 # converts degrees west to degrees east
mylon # answer = 262.8319
#%% same thing for lon
np.where(lon>262.8319)[0][0] # answer is 263
#%%
meand.shape # shape is 180, 360
#%%
meand[116,263] # doesn't work because it is ON LAND
#%%
plt.pcolormesh(lon,lat,meand)
plt.scatter(lon[263], lat[116])
plt.xlim(250,300)
plt.ylim(20,40)
# yep. it is on land.
#%% #%% move over one to offshore
plt.pcolormesh(lon,lat,meand)
plt.scatter(lon[264], lat[116])
plt.xlim(250,300)
plt.ylim(20,40)
# now we are in the water
#%%  double checks that we are in the water by giving a SST (24.422651 C)
meand[116,264] 
#%%
"""
Exercises From Lecture
"""
#%%
""" Exercise: Make a line plot of the monthly temperatures at this location """
# 1. Which month is the hottest? How hot is it?
# 2. Which is it coldest? How cold is it?
# 3. Now use max and min functions with np.where to find the above values
#%% Plotting
SPI = lon[264], lat[116]
#%%
spi = sst[:, 116, 264]
#%%
plt.plot(spi)
plt.xlabel('Months')
plt.ylabel('Temp ($^o$C)')
#%%
np.max(spi)
#%%
"""
1. Which month is the hottest?  September, at 28.368649 C
"""
#%%
np.min(spi)
#%%
"""
2. Which month is the coldest?   February, at 20.171156 C
"""
#%%
np.where(spi>=28.368649)
#%%
np.where(spi<=20.171156)
#%%
""" Exercise: In words, what is different about the arange and linspace x arrays? """
#%%
'''
The difference is that arange uses a step size while linspace uses the number
of samples to create its ndarray.
Additionally, arange struggles to use steps that are non-integers, such as 0.5.
Python help encourages you to use linspace instead.
'''
#%%
""" Exercies: Could we use x and y to define z above? Why or why not? """
#%%
'''
I believe with some serious maths you could probably figure out what z is based
on little x and little y, but I am honestly not sure.
'''
#%%
""" Exercise: Do exercise 3 and 4 above using meshgrids instead of for loops """

#%% Exercise 3
#%%
Lon, Lat = np.meshgrid(lon, lat)
#%%
plt.pcolormesh(Lat)
plt.colorbar()
#%%
plt.pcolormesh(Lon)
plt.colorbar()
#%% Masking the Tropics
mask1 = (Lat >= 23.4368) | (Lat <= -23.4368) # Sets all values below this one
#%%
trop = np.ma.masked_where(mask1, meand) # masks mask3 over meand
#%%
plt.figure(figsize=(15,3), dpi=70) # Sets Figure Size
im2 = plt.pcolormesh(lon, lat, trop, cmap='plasma', vmin=0, vmax=30) # Plots and Sets Colors
plt.ylim(-22, 23.4368) # Corrects the 'y' limits to remove white space
plt.suptitle('Annual Mean SST ($^o$C) for the Tropics, 1980', fontsize=20)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.colorbar(im2, orientation='vertical', pad=0.05, fraction=0.05, extend="both")
#%% Exercise 4
Lon, Lat = np.meshgrid(lon, lat)
#%%
plt.pcolormesh(Lat)
plt.colorbar()
#%%
plt.pcolormesh(Lon)
plt.colorbar()
#%% Masking the Tropics
mask2 = (Lat <= 23.4368) & (Lat >= -23.4368) # Just the opposite of mask 3
#%%
out = np.ma.masked_where(mask2, meand) # masks mask3 over meand
#%%
im2 = plt.pcolormesh(lon, lat, out, cmap='plasma', vmin=0, vmax=30) # Plots and Sets Colors
plt.suptitle('Annual Mean SST ($^o$C) for Areas Outside the Tropics, 1980', fontsize=20)
plt.xlabel('Longitude', fontsize=16)
plt.ylabel('Latitude', fontsize=16)
plt.colorbar(im2, orientation='vertical', pad=0.05, fraction=0.05, extend="both")