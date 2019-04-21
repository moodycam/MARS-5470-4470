# -*- coding: utf-8 -*-
"""
Created on Sat Apr 13 11:20:06 2019

@author: tomyamashita
"""

#%% Required packages

import mk_test
import pysal as ps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd
import gdal

#%% Create meshgrid from road file


SH100 = gpd.read_file("SH100_Transect.shp")

xmin = np.array(SH100.bounds.minx)
xmax = np.array(SH100.bounds.maxx)
ymin = np.array(SH100.bounds.miny)
ymax = np.array(SH100.bounds.maxy)

x = np.arange(xmin, xmax, 100)
y = np.arange(ymin, ymax, 100)

X, Y = np.meshgrid(x, y, indexing = 'xy')

coords = np.dstack((X, Y))
coords

coords_flat = coords.reshape((-1,2))

#%% 

SH100_buff = gpd.read_file("SH100_Buffer_300m.shp")

plt.pcolormesh(X)

SH100_buff.plot()


#%% 

cube = xr.open_dataset("cube.nc")

cube
x = cube.x
y = cube.y
occurrence = cube["OCCURRENCE_SUM_ZEROS"]
zscore = cube["COUNT_TREND_ZSCORE"]

plt.pcolormesh(lon, lat, zscore)

fig, ax = plt.subplots(ncols = 1, figsize = (10,5))
ax.pcolormesh(lon, lat, occurrence[0])

#%% Plots of mortality occurrence by time period

fig, ax = plt.subplots(figsize =(14,8), ncols=7, nrows=2, sharex = True, sharey = True)
 
n = 0
for x in range(0,2):
    for y in range(0,7):
        im = ax[x,y].pcolormesh(lon, lat, occurrence[n], cmap='Reds', vmin = 0, vmax = 7)
        n += 1 # Counter for the data referecne
        
#%%

fig, ax = plt.subplots(figsize = (15,6))
ax.pcolormesh(x, y, occurrence[0], cmap = "Reds", vmin = 0, vmax = 5)
#ax.add_patch(SH100_buff)


#%% 

cube = xr.open_dataset("cube.nc")
print(cube)

x = cube.x
y = cube.y

x_mesh = np.array(x)
y_mesh = np.array(y)

X, Y = np.meshgrid(x_mesh, y_mesh, indexing = 'xy')
coords = np.dstack((X, Y))

fig, ax = plt.subplots(ncols = 2, figsize = (15,5))
ax[0].pcolormesh(X)
ax[1].pcolormesh(Y)


#%% Creating mask based on shapefile. This was actually successful unlike most of the things above and last week

# Import stuff...
import fiona
from shapely.geometry import Polygon, Point
import xarray as xr
import numpy as np
import matplotlib.pyplot as plt

# Open the shapefile and modify it to do the necessary things. I don't know what some of the things are and why they're needed
file = fiona.open("SH100_Buffer_100m.shp")  # Converts the shapefile into a collection datatype
pol = file.next()  # Converts the shapefile collection into a dictionary datatype
poly_data = pol["geometry"]["coordinates"][0]  # File with the geometry of the shapefile
poly = Polygon(poly_data)  # Creates a new polygon from the shapefile geometry

# Open a netcdf file created from ArcGIS
cube = xr.open_dataset("cube.nc")

# X and Y coordinates (X is equivalent to longitude, Y is equivalent to latitude)
x = cube.x
y = cube.y

# Convert x and y into numpy arrays
xx = np.array(x)
yy = np.array(y)

# Create a meshgrid from the x/y coordinates arrays
X, Y = np.meshgrid(xx, yy, indexing = 'xy')
coords = np.dstack((X, Y))

# Create array with the correct shape but with all zeros in it
sh = (len(yy), len(xx))  # This is the shape of the data
mask = np.zeros(sh)

# Check the shape of the coords file (X/Y meshgrid of coordinates)
coords.shape
coords.shape[0]
coords.shape[1]

# Create loop that checks if the points in the grid are within the SH100 polygon
for x in range(0, coords.shape[0]):
    for y in range(0, coords.shape[1]):
        mask[x,y] = np.array([poly.contains(Point(coords[x,y,:]))])

# Plot the mask
fig, ax = plt.subplots(figsize = (16,6))
m = ax.pcolormesh(xx, yy, mask, cmap = "Blues")
ax.set_title("Masked SH100", fontsize = 18)
ax.set_xlabel("East-West coordinates (X)", fontsize = 14)
ax.set_ylabel("North-South coordinates (Y)", fontsize = 14)
plt.colorbar(m)

# Ignore this stuff
"""
#for i in range(0, len(xx),1):
    #points[i] = np.array([xx[i], yy[i]])

#mask = np.array([poly.contains(Point(x, y)) for x, y in points])
#poly.contains(Point(x,y)) for x, y in points
#points
#points.shape
#mask = np.array([poly.contains(Point(x, y)) for x, y in coords])

# keep lognitude constant and check if inside polygon
# poly is polygon from a shapefile (buffer around highway)
# contains
mask = np.array([poly.contains(Point(coords[:,0,:][x])) for x in range(0,len(coords[:,0,:]))])

# Need to fix the for loop so it loops over x and y values using coords.shape[0]
# where x = coords.shape[0] and y = coords.shape[1]

for x in range(0, len(coords)):
    for y in range(0, len(coords[x])):
        mask2 = np.array([poly.contains(Point(coords[x , y,:]))])
"""

#%% Testing converting NaNs to 0s. This also seems to work but its not clean or simple (See below)

occurrence = np.array(cube["OCCURRENCE_SUM_ZEROS"])

# Convert all nan values to 0s
occur = np.nan_to_num(occurrence)

# Take the first axis only
occurrence0 = occur[0]

# Add the occurrence and the mask so values inside the mask are no longer 0
occurrence01 = occurrence0 + mask

# Reconvert the 0 values to NaNs
occurrence01[occurrence01==0] = np.nan

# Substract the mask again so occurrence values are correct again
occurrence011 = occurrence01 - mask

# This needs to go into the netcdf file so the analysis can be run
# Also needs to be iterated through all values of occurrence

plt.pcolormesh(xx, yy, occurrence011, cmap = "Reds")


#%% Making the for loop for the occurrence thing from above. This works

occurrence = np.array(cube["OCCURRENCE_SUM_ZEROS"])

occur = np.nan_to_num(occurrence)

i = 0
for i in range(0, occur.shape[0]):
    occur_temp = occur[i] + mask
    occur_temp[occur_temp==0] = np.nan
    occur_temp = occur_temp - mask
    occur[i] = occur_temp
    i += 1

print(occurrence[0])
print(occur[0])

colormap = "seismic"  # Set the colormap for the figures
facecolor = "mintcream"  # Set the background color of the plot

# Plot these things and compare original to new
fig, ax = plt.subplots(nrows = 2, figsize = (15,10), sharex = True)
fig.suptitle('Comparison of original and updated data', fontsize = 20)
ax[0].set_facecolor(facecolor)
ax[0].pcolormesh(xx, yy, occurrence[0], cmap = colormap)
ax[0].set_title("Original", fontsize = 18)
ax[0].set_ylabel("North-South Direction (Y)", fontsize = 14)
ax[1].set_facecolor(facecolor)
ax[1].pcolormesh(xx, yy, occur[0], cmap = colormap)
ax[1].set_title("New", fontsize = 18)
ax[1].set_ylabel("North-South Direction (Y)", fontsize = 14)
ax[1].set_xlabel("East-West Direction (X)", fontsize = 14)

"""
Next step seems to be to figure out how to add this output (occur) to the netcdf file
so ArcGIS things can be run on it
"""


#%% This version doesn't work
"""
occurrence = np.array(cube["OCCURRENCE_SUM_ZEROS"])

i = 0
for i in range(0, occur.shape[0]):
    occur = np.nan_to_num(occurrence)
    occur_temp = occur[i] + mask
    occur_temp[occur_temp==0] = np.nan
    occur_temp = occur_temp - mask
    occur[i] = occur_temp
    i += 1

"""

#%%  Psuedo code for next steps

"""
import arcpy

RUN create space time cube tool through arcpy

Input my intermediary code to replace nans with 0

RUN emerging hotspot analysis through arcpy

"""

#%% Saving the output of this code to the netcdf file

"""
DON'T use this. It deleted data
cube["OCCURRENCE_SUM_ZEROS"][0] = occur[0]

cube["OCCURENCE_SUM_ZEROS"]
"""

"""
It looks like I may have to rewrite the above code using netCDF4 instead of xarray
Nan issue will have to be dealt with differently because netCDF4 and xarray handle NaNs differently
"""

# Full code without plots

import fiona
from shapely.geometry import Polygon, Point
import numpy as np
import matplotlib.pyplot as plt
import netCDF4


# When dealing with the arcgis stuff, importing the netcdf file may be different
cube = netCDF4.Dataset("cube.nc", 'r+') # Need to open with 'r+' so that the file can be edited
occurrence = np.array(cube["OCCURRENCE_SUM_ZEROS"])
print(occurrence)

# Testing code. This works
#test2 = test_occur[0]+(mask)
#test2[test2==-9998] = 0

# X and Y coordinates are accessed in a different way...
x = np.array(cube["x"])
y = np.array(cube["y"])

## From the above code
# Open the shapefile and modify it to do the necessary things. I don't know what some of the things are and why they're needed
file = fiona.open("SH100_Buffer_100m.shp")  # Converts the shapefile into a collection datatype
pol = file.next()  # Converts the shapefile collection into a dictionary datatype
poly_data = pol["geometry"]["coordinates"][0]  # File with the geometry of the shapefile
poly = Polygon(poly_data)  # Creates a new polygon from the shapefile geometry

# Create a meshgrid from the x/y coordinates arrays
X, Y = np.meshgrid(x, y, indexing = 'xy')
coords = np.dstack((X, Y))

# Create array with the correct shape but with all zeros in it
sh = (len(y), len(x))  # This is the shape of the data
mask = np.zeros(sh)

# Check the shape of the coords file (X/Y meshgrid of coordinates)
coords.shape
coords.shape[0]
coords.shape[1]

# Create loop that checks if the points in the grid are within the SH100 polygon
for x in range(0, coords.shape[0]):
    for y in range(0, coords.shape[1]):
        mask[x,y] = np.array([poly.contains(Point(coords[x,y,:]))])

## End repeat code


# Redone for loop accounting for the fact that all real values are greater than 0 and all nans are -9999
for i in range(0, occurrence.shape[0]):
    occur_temp = occurrence[i] + mask
    occur_temp[occur_temp==-9998] = 0
    occurrence[i] = occur_temp
    i += 1 
print(occurrence)

cube['OCCURRENCE_SUM_ZEROS'][:] = occurrence
cube.close()


"""
Looks like there is also a mask on this dataset which I will have to change as well
This can just be replaced with the mask I created above
"""

#%% 

