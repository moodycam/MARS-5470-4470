#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 16:45:43 2019

@author: roryeggleston
"""

import geopandas as gpd
import matplotlib.pyplot as plt
#%%
file = '/Users/roryeggleston/Downloads/ICES_ecoregions/ICES_ecoregions_20171207_erase_ESRI.shp'
#%%
data = gpd.read_file(file)
#%%
type(data)
#%%
data.head()
#%%
data.plot()
#%%
data.crs
#%%
data['geometry'].head()
#%%
data_proj = data.copy()
#%%
data_proj['geometry'] = data_proj['geometry'].to_crs(epsg=3879)
#%%
data_proj['geometry'].head()
#%%
data.plot(markersize=6, color="indigo");
plt.title("Projection 1");
plt.tight_layout()
data_proj.plot(markersize=6, color="lavender");
plt.title("Projection 2");
plt.tight_layout()