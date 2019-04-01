#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 27 14:09:22 2019

@author: roryeggleston
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import geopandas as gpd
from descartes import PolygonPatch
#%%
#EXERCISE 1: No problems loading the necessary packages.
#%%
def load_shape_file(filepath):
    shpfile = gpd.read_file(filepath)
    return shpfile
#%%
shp = load_shape_file('/Users/roryeggleston/Downloads/LME66/LMEs66.shp')
shp.head()
#%%
shp
#%%
print(shp)
#%%
#EXERCISE 2
type(shp)
#the shapefile is a geopandas dataframe
#%%
from mpl_toolkits.basemap import Basemap
#%%
ax = plt.figure(figsize=(16,20), facecolor = 'w')
# plot lims
limN, limS, limE, limW = 84.,-80.,180,-180
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#BDA973', lake_color='#BDA973')
#%%
# latlon lines
ax = plt.figure(figsize=(16,20), facecolor = 'w')
limN, limS, limE, limW = 84.,-80.,180,-180
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#BDA973', lake_color='#BDA973')
parallels = np.arange(-90.,90,20.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
#meridians
meridians = np.arange(-180.,180.,20.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#%%
#EXCERISE 3
#A: 
ax = plt.figure(figsize=(16,20), facecolor = 'w')
limN, limS, limE, limW = 84.,-80.,180,-180
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='coral', lake_color='#BDA973')
parallels = np.arange(-90.,90,20.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
#meridians
meridians = np.arange(-180.,180.,20.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#%%
#B: 
ax = plt.figure(figsize=(16,20), facecolor = 'w')
limN, limS, limE, limW = 84.,-80.,180,-180
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='coral', lake_color='blue')
parallels = np.arange(-90.,90,20.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
#meridians
meridians = np.arange(-180.,180.,20.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#%%
#C:
ax = plt.figure(figsize=(16,20), facecolor = 'w')
limN, limS, limE, limW = 84.,-80.,180,-180
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='l')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='coral', lake_color='blue')
parallels = np.arange(-90.,90,20.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
#meridians
meridians = np.arange(-180.,180.,20.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#%%
#D:
ax = plt.figure(figsize=(12,16), facecolor = 'w')
limN, limS, limE, limW = 35.,15.,-80,-100
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='l')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='coral', lake_color='blue')
parallels = np.arange(15.,35,5.)
m.drawparallels(parallels,labels=[1,0,0,0],fontsize=10)
#meridians
meridians = np.arange(-180.,80.,5.)
m.drawmeridians(meridians,labels=[0,0,0,1],fontsize=10)
#%%
sppath = '/Users/roryeggleston/Downloads/LME66/LMEs66'
#%%
ax = plt.figure(figsize=(16,20), facecolor = 'w')
limN, limS, limE, limW = 84.,-80.,180,-180
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#d8b365', lake_color='w')

m.readshapefile(sppath, 'LME') # 2nd argument is name  for shapefile data inside the shapefile
# plot all the shapes in the shapefile
for info, shape in zip(m.LME_info, m.LME):
        x, y = zip(*shape) 
        m.plot(x, y, marker=None,color='k', linewidth = '2')
#%%
#EXERCISE 4
ax = plt.figure(figsize=(16,20), facecolor = 'w')
limN, limS, limE, limW = 84.,-80.,180,-180
m = Basemap(projection='cyl', llcrnrlon=limW, \
      urcrnrlon=limE, llcrnrlat=limS, urcrnrlat=limN, resolution='c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#d8b365', lake_color='w')

m.readshapefile(sppath, 'LME') # 2nd argument is name  for shapefile data inside the shapefile
# plot all the shapes in the shapefile
for info, shape in zip(m.LME_info, m.LME):
        x, y = zip(*shape) 
        m.plot(x, y, marker=None,color='purple', linewidth = '6')
#%%
def select_shape(shpfile, category, name):
    s = shpfile
    polygon = s[s[category] == name]
    polygon = polygon.geometry[:].unary_union
    return polygon
#%%
CalCS_shp = select_shape(shp, 'LME_NAME', 'California Current')
CalCS_shp
#%%
#EXERCISE 5
type(CalCS_shp)
#CalCS_shp is a Polygon, no shape specified
#%%
#EXERCISE 6
#Gulf of Alaska
GoA_shp = select_shape(shp, 'LME_NAME', 'Gulf of Alaska')
GoA_shp
#%%
#Labrador-Newfoundland
LaNf_shp = select_shape(shp, 'LME_NAME', 'Labrador - Newfoundland')
LaNf_shp
#%%
def lat_lon_formatter(ax):
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=16)
    
#this uses the above latlon formatter
def set_up_map(ax, x0, x1, y0, y1):
    # land overlay
    ax.add_feature(cfeature.LAND, facecolor='k')
    # zoom on  desired region
    ax.set_extent([x0, x1, y0, y1])
    
    #better ticks
    ax.set_xticks(np.arange(x0, x1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(y0, y1, 10), crs=ccrs.PlateCarree())
    lat_lon_formatter(ax)
#%%
f, ax = plt.subplots(ncols=2, figsize=(10,5),
                     subplot_kw=dict(projection=ccrs.PlateCarree())) #map projection from cartopy
set_up_map(ax[0], -140, -107, 20, 50)
set_up_map(ax[1], -140, -107, 20, 50)
#%%
#EXERCISE 7
#You can change the color of the land overlay in the set_up_map function by changing facecolor
def lat_lon_formatter(ax):
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=16)
def set_up_map(ax, x0, x1, y0, y1):
    ax.add_feature(cfeature.LAND, facecolor='Purple')
    ax.set_extent([x0, x1, y0, y1])
    ax.set_xticks(np.arange(x0, x1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(y0, y1, 10), crs=ccrs.PlateCarree())
    lat_lon_formatter(ax)
f, ax = plt.subplots(ncols=2, figsize=(10,5),
                     subplot_kw=dict(projection=ccrs.PlateCarree()))
set_up_map(ax[0], -140, -107, 20, 50)
set_up_map(ax[1], -140, -107, 20, 50)
#%%
from descartes import PolygonPatch
#%%
help(PolygonPatch)
#%%
f, ax = plt.subplots(ncols=2, figsize=(10,5),
                     subplot_kw=dict(projection=ccrs.PlateCarree()))
set_up_map(ax[0], -140, -107, 20, 50)
set_up_map(ax[1], -140, -107, 20, 50)
# add shapefile to map
ax[0].add_patch(PolygonPatch(CalCS_shp, fc='#add8e6'))
ax[1].add_patch(PolygonPatch(CalCS_shp, fc='None', ec='r', linewidth=2,
                             linestyle=':'))
#%%
fig= plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
set_up_map(ax, -140, -107, 20, 50)

# plot the shapefile in blue with a red dotted boundary
ax.add_patch(PolygonPatch(CalCS_shp, fc='None', ec='r', linewidth=2,
                             linestyle=':'))
ax.add_patch(PolygonPatch(CalCS_shp, fc='#add8e6', ec = 'None', alpha = 1, zorder = 0))
#%%
#EXERCISE 8
fig= plt.figure(figsize=(5,5))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
set_up_map(ax, -140, -107, 20, 50)

# plot the shapefile in blue with a red dotted boundary
ax.add_patch(PolygonPatch(CalCS_shp, fc='None', ec='r', linewidth=2,
                             linestyle=':'))
ax.add_patch(PolygonPatch(CalCS_shp, fc='#add8e6', alpha = 1, zorder = 10))
#Changing zorder to 10 and taking out the ec statement seems to make the coastal outline thinner and also overlays a black line over the red dashes.
#%%
#EXERCISE 9
GoC_shp = select_shape(shp, 'LME_NAME', 'Gulf of California')
GoC_shp
#%%
PCAC_shp = select_shape(shp, 'LME_NAME', 'Pacific Central-American Coastal')
PCAC_shp
#%%
HC_shp = select_shape(shp, 'LME_NAME', 'Humboldt Current')
HC_shp
#%%
fig= plt.figure(figsize=(7,15))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
set_up_map(ax, -170, -70, -60, 70)
ax.add_patch(PolygonPatch(CalCS_shp, fc='blue', ec = 'r', alpha = 1, zorder = 0))
ax.add_patch(PolygonPatch(GoA_shp, fc='turquoise', ec = 'r', alpha = 1, zorder = 0))
ax.add_patch(PolygonPatch(GoC_shp, fc='teal', ec = 'r', alpha = 1, zorder = 0))
ax.add_patch(PolygonPatch(PCAC_shp, fc='orange', ec = 'r', alpha = 1, zorder = 0))
ax.add_patch(PolygonPatch(HC_shp, fc='yellow', ec = 'r', alpha = 1, zorder = 0))