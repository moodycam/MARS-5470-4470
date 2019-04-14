#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:28:02 2019

@author: roryeggleston
"""
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import pandas as pd
import xarray as xr
import netCDF4 as nc
from sklearn.cluster import KMeans
#%%
file = '/Volumes/MY PASSPORT/CORT_BABBLE_DATA_040819.xlsx'
babble = pd.read_excel(file)
#%%
babble.shape
#%%
#TRYING TO DO K-MEANS BEFORE PCA, NOT GONNA WORK FOR MY DATA
kmeans_vars = babble[[("Delta Time (s)"), ("IQR BW (Hz)"), ("Avg Entropy (bits)"), ("Center Freq (Hz)"), ("Freq 5% Rel."), ("Freq 95% Rel."), ("PFC Avg Slope (Hz/ms)")]]
#%%
kmeans = KMeans(n_clusters=7)
kmeans.fit(kmeans_vars)
#%%
labels = kmeans.predict(kmeans_vars)
centroids = kmeans.cluster_centers_
#%%
colmap = {1: 'red', 2: 'green', 3: 'blue', 4: 'purple', 5: 'turquoise', 6: 'coral', 7: 'orange'}
#%%
fig = plt.figure(figsize=(5, 5))

colors = map(lambda x: colmap[x+1], labels)

plt.scatter(kmeans_vars["Delta Time (s)"], kmeans_vars["IQR BW (Hz)"], kmeans_vars["Avg Entropy (bits)"], kmeans_vars["Center Freq (Hz)"], kmeans_vars["Freq 5% Rel."], kmeans_vars["Freq 95% Rel."], kmeans_vars["PFC Avg Slope (Hz/ms)"], color=colors, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=colmap[idx+1])
plt.xlim(0, 80)
plt.ylim(0, 80)
plt.show()
#%%
#NEXT TIME WILL BE WORKING ON DOING PCA FIRST
