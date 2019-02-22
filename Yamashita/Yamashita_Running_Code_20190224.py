# -*- coding: utf-8 -*-
"""
Created on Thu Feb 21 22:43:46 2019

@author: tjyamashta_dell
"""

#%% 2/18/2019 Lecture

"""
Challenge Problems

Using the 1980 monthly SST netcdf file we used last time:

    1. Take the annual mean and plot it with the correct lat and lon. 
        Use a different colormap than the default. 
        Add axes labels, a colorbar and title. 
        In the title, include what the plotted variable is and the units.

    2. Take seasonal averages: DJF, etc. 
        Plot the four seasons in a 2x2 plot of subplots, 
        label each plot and put on one colorbar for all the plots, ends extended, with a min value of 0 and a max of 30. 
        Make a descriptive supertitle (suptitle) for all the plots. 
        Would it be correct to label these plots Winter, etc?

    3. Mask out regions outside of the tropics (google definition of tropics) and 
        plot the annual mean again. Adjust the font size of your axes labels and title. 
        Adjust the axis limits to be constrained to the data (i.e. no whitespace).

    4. Mask out the tropics and plot again.

    5. (Advanced) Find the grid cell closest to us. 
        How big is this cell in square km and miles? 
        Plot the seasonal cycle of SST. 
        What are our local seasonal averages, according to this model? 
        Annual Mean? Are they realistic? 
        What data sources could you use to validate this?

"""

#%% Importing things for problems

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
dataset = xr.open_dataset("CESM.003.SST.1980.nc")

#%% Problem 1

data1 = dataset
lat = np.array(data1.lat)
lon = np.array(data1.lon)
SST = np.array(data1.SST)
mean_SST = np.mean(SST, axis = 0)

fig, ax = plt.subplots(figsize = (8,6))
p = ax.pcolormesh(lon, lat, mean_SST, cmap = "coolwarm")
plt.colorbar(p, extend="both")
plt.xlabel("Longitude"), plt.ylabel("Latitude")
plt.title("Mean Sea Surface Temperature", fontsize = 16)

#%% Problem 2

# Note: The number of brackets in a numpy array is related to the number of dimensions in the array

data2 = dataset
lat = np.array(data2.lat)
lon = np.array(data2.lon)
SST = np.array(data2.SST)
# Can also use: 
# winter = SST[(0,1,11),:,:]
winter = np.array([SST[0], SST[1], SST[11]])
spring = np.array([SST[2], SST[3], SST[4]])
summer = np.array([SST[5], SST[6], SST[7]])
fall = np.array([SST[8], SST[9], SST[10]])

mean_winter = np.mean(winter, axis = 0)
mean_spring = np.mean(spring, axis = 0)
mean_summer = np.mean(summer, axis = 0)
mean_fall = np.mean(fall, axis = 0)

fig, ax = plt.subplots(figsize = (12,8), ncols = 2, nrows = 2, sharex = True, sharey = True)
wi = ax[0,0].pcolormesh(lon, lat, mean_winter, cmap = "coolwarm", vmin = 0, vmax = 30)
sp = ax[1,0].pcolormesh(lon, lat, mean_spring, cmap = "coolwarm", vmin = 0, vmax = 30)
su = ax[0,1].pcolormesh(lon, lat, mean_summer, cmap = "coolwarm", vmin = 0, vmax = 30)
fa = ax[1,1].pcolormesh(lon, lat, mean_fall, cmap = "coolwarm", vmin = 0, vmax = 30)
plt.colorbar(su, ax = ax, extend = "both")
ax[0,0].set_title("December, January, February")
ax[0,1].set_title("March, April, May")
ax[1,0].set_title("June, July, August")
ax[1,1].set_title("September, October, November")
fig.text(0.45, 0.05, "Longitude", ha = "center", va = "center")
fig.text(0.05, 0.5, "Latitude", ha = "center", va = "center", rotation = "vertical")
fig.suptitle("Mean Sea Surface Temperature by Season", x=0.45, fontsize = 16)

#%% Problem 3

data3 = dataset
lat = np.array(data3.lat)
lon = np.array(data3.lon)
SST = np.array(data3.SST)
mean_SST = np.mean(SST, axis = 0)

# (Not sure if this is right. It's not...) tropics_mask = ((lat < 23.5) & (lat > -23.5))
# SST_masked = np.ma.masked_where(tropics_mask, (SST.shape[0],1))
"""
mean_SST has shape (180,360)
need to remove those values greater than (90+23.5) and less than (90-23.5)

"""
temp_mask1 = ((lat > (0 + 23.4368)) | (lat < (0 - 23.4368)))
# temp_mask2 = np.arange(1,361)

# tropics[:,0] = temp_mask1 (Test Code)

n = 0
tropics = np.zeros((180,360))
for x in range(0,360):
    tropics[:,x] = temp_mask1 
    n += 1


SST_mask = np.ma.masked_where(tropics, mean_SST)

fig, ax = plt.subplots(figsize = (12,4))
p = ax.pcolormesh(lon, lat, SST_mask, cmap = "coolwarm")
plt.colorbar(p, extend = "both")
plt.xlabel("longitude", fontsize = 14)
plt.ylabel("Tropical Latitudes", fontsize = 14)
plt.ylim(-23.4368, 23.4368)
plt.title("Mean Sea Surface Tempearture for the Tropics", fontsize = 18)

#%% Problem 4

data4 = dataset
lat = np.array(data4.lat)
lon = np.array(data4.lon)
SST = np.array(data4.SST)
mean_SST = np.mean(SST, axis = 0)

# (Not sure if this is right. It's not...) tropics_mask = ((lat < 23.5) & (lat > -23.5))
# SST_masked = np.ma.masked_where(tropics_mask, (SST.shape[0],1))
"""
mean_SST has shape (180,360)
need to remove those values greater than (90+23.5) and less than (90-23.5)

"""
temp_mask1 = ((lat < (0 + 23.4368)) & (lat > (0 - 23.4368)))
#temp_mask2 = np.arange(1,361)

# tropics[:,0] = temp_mask1 (Test Code)

n = 0
tropics = np.zeros((180,360))
for x in range(0,360):
    tropics[:,x] = temp_mask1 
    n += 1


SST_mask = np.ma.masked_where(tropics, mean_SST)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 50
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['lines.linewidth'] = 2.0

fig, ax = plt.subplots(figsize = (12,8))
p = ax.pcolormesh(lon, lat, SST_mask, cmap = "coolwarm")
plt.colorbar(p, extend = "both")
plt.xlabel("longitude", fontsize = 14)
plt.ylabel("Tropical Latitudes", fontsize = 14)
plt.title("Mean Sea Surface Tempearture for the Tropics", fontsize = 18)

#%% Problem 5

# Brownsville's location: 
# Latitude = 25.930278
# Longitude = -97.484444

data5 = dataset
lat = np.array(data5.lat)
lon = np.array(data5.lon)
SST = np.array(data5.SST)
mean_SST = np.mean(SST, axis = 0)

"""
Find the cell containing our location
    Some reference to call "cell containing these coordinates [:,local lat,local lon]
Mask everything that isn't that cell
    (SST <> the cell of interest)
Area is calculated as distance (lat) * distance (lon) 
    where lat distance never varies and lon distance varies with latitude
Plot seasonal cycle as line graph?

"""

local_lat = np.array([25.930278])
local_lon = np.array([180 + 97.484444])

# (This isn't right) local_coord = np.array([local_lat, local_lon])

# This works but doesn't seem right or simple. Also here is a cell only containing land
here = SST[:,90+26,277]
# Moving closer to the ocean
near1 = SST[:,90+26,278]
near2 = SST[:,90+26,279]
near3 = SST[:,90+26,280]
near4 = SST[:,90+26,281]

#fig, ax = plt.subplots(figsize = (12,6), ncols = 1, nrows = 1)
#map1 = ax.pcolormesh(lon, lat, mean_SST, cmap = "seismic")
#plt.colorbar(map1, extend = "both")
#ax.plot(near4)


fig, ax = plt.subplots(figsize = (12,6), ncols = 2)
map1 = ax[0].pcolormesh(lon, lat, mean_SST, cmap="seismic", vmin=0, vmax=25)
plt.colorbar(map1, extend = "both")
line1 = ax[1].plot(near4)

# This was finished on 2/20

#%% 2/20/2019 Lecture 6.2

# Finding data

# Brownsville's location: 
# Latitude = 25.930278
# Longitude = -97.484444

# SPI location
# latitude = 26.1118
# Longitude = -97.1681

data5 = dataset
lat = np.array(data5.lat)
lon = np.array(data5.lon)
SST = np.array(data5.SST)
mean_SST = np.mean(SST, axis = 0)

latind = np.where(lat > 26.1118)

ind = np.where(mean_SST > 20)
xind, yind = np.where(mean_SST > 20)
print(xind)
print(yind)
mean_SST[53,21] # To check that the value associated with particular x and y coordinates fulfil the criteria
xind?

# Use where function to find values that fit a criteria from an array
# The bracketed values are referencing a specific value

# Double reference is because the object created by np.where creates a tuple of arrays
# There is a tuple with size x (= size of array you are indexing) 
# each value of x is an array so you need to reference the value in the array to get a value of latitude/longitude

# so you have to access the a data point which is inside an array which inside another array
# so refence the array you want, then reference the data point within the array
local_lat = np.where(lat>25.930278)[0][0]
local_lon = np.where(lon>(360-97.48444))[0][0]

plt.pcolormesh(lon, lat, mean_SST)
plt.scatter(lon[local_lon+2], lat[local_lat], color = "black")
plt.xlim(260, 270)
plt.ylim(25, 30)

# Will be coming back to this

#%% Loading data from excel

import pandas as pd

# Can use the r'(file path)' instead of using the double slashes

mortalities = pd.read_excel("Sample_datasets.xlsx", sheet_name = "Mortalities")

date = np.array(mortalities.Date)
species = np.array(mortalities.Common)

#%% Meshes

x = np.linspace(-np.pi*2, np.pi*2, 50)
x = np.arange(-np.pi*2, np.pi*2, .5)

y1 = np.sin(x)
y2 = 0.1*x**2 -1

plt.plot(x,x,'k.')
plt.plot(x,y1)
plt.plot(x,y2)

#%% Plotting 2D functions

x = np.linspace(-np.pi*2, np.pi*2, 50)
y = np.linspace(-1,1,50)
# Create a grid array from 1D arrays
X,Y = np.meshgrid(x,y)

"""
x is 1D array contains 50 points changing from -2pi - 2pi 
y is 1D array contains 50 points changing from -1 - 1
meshgrid repeats the 1D array of x for each value of y and vice versa
the output is X and Y which are 2D arrays that contain repeated values of x or y over each value of y or x
"""

print(X)
print(Y)
print(x == X)
print(y == Y)
print(x.shape)
print(X.shape)
print(y.shape)
print(Y.shape)

plt.pcolormesh(X)
plt.colorbar()
plt.pcolormesh(Y)
plt.colorbar()

#%% Evaluating a function

z = np.sin(X*Y)
plt.pcolormesh(x,y,z)
plt.colorbar()


# Pcolormesh can recognize axes of 2D arrays
z = np.sin(X*Y)
plt.pcolormesh(X,Y,z)
plt.colorbar()

#%% Exercises

data5 = dataset
lat = np.array(data5.lat)
lon = np.array(data5.lon)
SST = np.array(data5.SST)
mean_SST = np.mean(SST, axis = 0)

local_lat = np.where(lat>25.930278)[0][0]
local_lon = np.where(lon>(360-97.48444))[0][0]

near1 = SST[:, local_lat,local_lon+2]

fig, ax = plt.subplots(figsize = (12,6), ncols=2)
map1 = ax[0].pcolormesh(lon, lat, mean_SST, cmap = "coolwarm")
plt.colorbar(map1, extend = "both")
ax[0].scatter(lon[local_lon+2], lat[local_lat], color = "black")
ax[0].set_xlim(260, 270)
ax[0].set_ylim(25, 30)
ax[1].plot(near1)


np.where(near1 == max(near1))[0][0]

# Not sure how to return the index number of the month with the max/min temperature to complete this print statement
print("The max temperature is in month " + str(np.where(near1 == max(near1))[0][0] +1) + " and the temperature is " + str(max(near1)))
print("The min temperature is in month " + str(np.where(near1 == min(near1))[0][0] +1) + " and the temperature is " + str(min(near1)))

#%% Exercise 2

"""
linspace creates an array where you specify the start and end points and the number of evenly spaced values between them
arange creates an array where you specify the start and end points and the distance between each point
"""

#%% Exercise 3

x = np.linspace(-np.pi*2, np.pi*2, 50)
y = np.linspace(-1,1,50)
# Create a grid array from 1D arrays
X,Y = np.meshgrid(x,y)

Z = np.sin(X*Y)
z = np.sin(x*y)

plt.plot(x,x, color = "blue")
plt.plot(x,y, color = "red")
plt.plot(x,z, color = "green")

"""
Using x and y instead of X and Y, z becomes a 1D array rather than a multi-dimensional array
In theory, yes you could use x and y but you would only get a 1D array out of it so I guess it would depend on what you are trying to produce
"""

#%% Redoing previous ex 3 and 4

data3 = dataset
lat = np.array(data3.lat)
lon = np.array(data3.lon)
SST = np.array(data3.SST)
mean_SST = np.mean(SST, axis = 0)

"""
mean_SST has shape (180,360)
need to remove those values greater than (90+23.5) and less than (90-23.5)
"""

# Creating the meshgrid 
temp_mask1 = ((lat > (0 + 23.4368)) | (lat < (0 - 23.4368)))
lon_temp = np.zeros(360)
lon_mesh, lat_mesh = np.meshgrid(lon_temp, temp_mask1)

SST_mask = np.ma.masked_where(lat_mesh, mean_SST)

fig, ax = plt.subplots(figsize = (12,4))
p = ax.pcolormesh(lon, lat, SST_mask, cmap = "coolwarm")
plt.colorbar(p, extend = "both")
plt.xlabel("longitude", fontsize = 14)
plt.ylabel("Tropical Latitudes", fontsize = 14)
plt.ylim(-23.4368, 23.4368)
plt.title("Mean Sea Surface Tempearture for the Tropics", fontsize = 18)

#%% Problem 4

data4 = dataset
lat = np.array(data4.lat)
lon = np.array(data4.lon)
SST = np.array(data4.SST)
mean_SST = np.mean(SST, axis = 0)

"""
mean_SST has shape (180,360)
need to remove those values greater than (90+23.5) and less than (90-23.5)
"""

temp_mask1 = ((lat < (0 + 23.4368)) & (lat > (0 - 23.4368)))
lon_temp = np.zeros(360)
lon_mesh, lat_mesh = np.meshgrid(lon_temp, temp_mask1)

SST_mask = np.ma.masked_where(lat_mesh, mean_SST)

import matplotlib as mpl
mpl.rcParams['figure.dpi'] = 50
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['lines.linewidth'] = 2.0

fig, ax = plt.subplots(figsize = (12,8))
p = ax.pcolormesh(lon, lat, SST_mask, cmap = "coolwarm")
plt.colorbar(p, extend = "both")
plt.xlabel("longitude", fontsize = 14)
plt.ylabel("Tropical Latitudes", fontsize = 14)
plt.title("Mean Sea Surface Tempearture for the Tropics", fontsize = 18)

#%% Back to importing excel data

import pandas as pd

# Can use the r'(file path)' instead of using the double slashes

mortalities = pd.read_excel("Sample_datasets.xlsx", sheet_name = "Mortalities")

date = np.array(mortalities.Date)
road = np.array(mortalities.Road)
species = np.array(mortalities.Common)

# Not sure how to work with categorical data
# Example create a bar graph of sums of species because sum function only works with numbers
# (Yes I did test this. I just erased it)
