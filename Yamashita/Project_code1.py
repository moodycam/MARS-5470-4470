# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:31:57 2019

@author: tjyamashta_dell
"""

#%% Import required modules
import mk_test
import pysal as ps
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import geopandas as gpd
import pandas as pd

#%% load and view the ArcGIS netCDF file

testcube = xr.open_dataset("cube.nc")

print(testcube)

zscore = testcube["COUNT_TREND_ZSCORE"]
lat = testcube.lat
lon = testcube.lon

# Everything has the same shape
print(zscore.shape)
print(lat.shape)
print(lon.shape)

plt.pcolormesh(lon, lat, zscore, cmap = "bwr")


#%% Continuing to look at this

mortalities = np.array(testcube["OCCURRENCE_SUM_ZEROS"])

print(mortalities.shape)
mortalities

plt.pcolormesh(lon, lat, mortalities[0], cmap = 'bwr')


#%% Test hot spot analysis (Ignore this)
"""
# Need to figure out where the functions are because documentation has changed
mortalities0 = mortalities[0]

# Need to create a weights file before running the hot spot analysis
# ArcGIS version uses a kernel weights calculation

w = ps.weights.Distance.DistanceBand(data = mortalities0, threshold = 100., binary = False)

ps.explore.esda.G_Local(mortalities[0])
"""

#%% Trying it with shapefile

shp = "Mortalities_SH100.shp"

mort_shape = gpd.read_file(shp)

total_morts = mort_shape[mort_shape["Occurrence"]==1]
mammals = total_morts[total_morts["Class"] == "Mammalia"]
reptiles = total_morts[total_morts["Class"] == "Reptilia"]
amphibians = total_morts[total_morts["Class"] == "Amphibia"]

land = pd.concat([mammals, reptiles, amphibians])
morts = land["Occurrence"]

thresh = ps.min_threshold_dist_from_shapefile(shp)
print(thresh)

wt = ps.threshold_binaryW_from_shapefile(shp, 90)
wt.min_neighbors
wt.histogram
ps.G_Local(morts, wt)

#%% Trying it again

shp = "Mortalities_SH100.shp"

mortality = gpd.read_file(shp)

total_morts = mort_shape[mort_shape["Occurrence"]==1]
mammals = total_morts[total_morts["Class"] == "Mammalia"]
reptiles = total_morts[total_morts["Class"] == "Reptilia"]
amphibians = total_morts[total_morts["Class"] == "Amphibia"]

land = pd.concat([mammals, reptiles, amphibians])

gpd.GeoDataFrame.to_file(land, "land_mortality.shp")

#%% 

# This can't be used because the data hasn't been broken into blocks yet

land2 = "land_mortality.shp"
land_df = gpd.read_file(land2)
#occurrence = np.array(land_df["Occurrence"])

thresh = ps.min_threshold_dist_from_shapefile(land2)
print(thresh)

wt = ps.threshold_binaryW_from_shapefile(land2, 300)

HS = ps.G_Local(land_df["Index"], wt)

#%% Back to the original netCDF file 

test = xr.open_dataset("cube.nc")
print(test)

mortalities1 = np.array(test["OCCURRENCE_SUM_ZEROS"])[1]
print(mortalities1)
print(mortalities1.shape)

thresh = ps.min_threshold_distance_from_array(mortalities1)
wt = ps.DistanceBand(mortalities1, threshold = 300)

#%% Using a new function to open the files

"""
# Open dataset in pysal
land = ps.open("land_mortality.shp")

# View the header of the dataset (mostly metadata)
land.header

# View specific data by rows
land.by_row(1)

# Use the read function to get all the shapes in the shapefile
all_land = land.read()

# Identify the number of shapes in the file
len(all_land)

# Show the coordinates of the first point
all_land[0]

all_land[0]

land_again = ps.pdio.read_files("Mortalities.SH100.shp")
"""

#land = ps.core.IOHandlers.arcgis_txt.ArcGISTextIO("Mortalities_SH100.shp")
#all_land = land.read()

Morts = ps.pdio.read_files("Mortalities_SH100.shp")
Morts.head()

First = Morts[Morts["Occurrence"]==1]
Mammals = First[First["Class"]=="Mammalia"]
Reptiles = First[First["Class"]=="Reptilia"]
Amphibians = First[First["Class"]=="Amphibia"]

land_view = pd.concat([Mammals, Reptiles, Amphibians])

# Land is no longer a shapefile because it is a pandas dataframe
threshold = ps.min_threshold_dist_from_shapefile(land)

land = ps.open("Land_Mortality.shp")
all_land = land.read()

threshold = ps.min_threshold_dist_from_shapefile(land)
weight = ps.DistanceBand(land, threshold = 300)
weight.alpha
weight.histogram
weight.n

HS = ps.G_Local(land.Occurrence.values, weight, transform = 'B')


#%% Tutorial example because it has things that don't make sense in their text

ps.examples.available()

sample = ps.pdio.read_files("texas.shp")
# In the tutorial, HR90 is a column in the dataset

#%% Trying again

land_view = ps.pdio.read_files("Mortalities_SH100_Land.shp")
land = ps.open("Mortalities_SH100_land.shp")

weight = ps.DistanceBand(land, threshold = 300)

weight.min_neighbors
weight.n

# This produces the NaNs and infs output so something is wrong
HS = ps.G_Local(land_view.Occurrence.values, weight)


#%% Back to the netCDF cube

cube = xr.open_dataset("cube.nc")

Time1 = np.array(cube.OCCURRENCE_SUM_ZEROS[0])

# This cannot create a weights file because there are too many NaNs
weight = ps.Kernel(Time1, function = 'gaussian')

#%% Creating grids and masked arrays

land_view.UTM_Y.min()
land_view.UTM_Y.max()

x = np.arange(land_view.UTM_X.min(), land_view.UTM_X.max(), 100)
y = np.arange(land_view.UTM_Y.min(), land_view.UTM_Y.max(), 100)
X, Y = np.meshgrid(x, y)
X2 = X.reshape((np.prod(X.shape),))
Y2 = Y.reshape((np.prod(Y.shape),))
coords = zip(X2,Y2)


"""
create polygon of roads
import shapefile into python
identify which points in a meshgrid are within the polygon
"""

land_view

#%% Create netCDF file

