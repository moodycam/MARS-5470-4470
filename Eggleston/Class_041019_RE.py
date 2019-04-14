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
kmeans_vars = babble[[("N"), ("Treatment"), ("Nest"), ("Delta Time (s)"), ("IQR BW (Hz)"), ("Avg Entropy (bits)"), ("Center Freq (Hz)"), ("Freq 5% Rel."), ("Freq 95% Rel."), ("PFC Avg Slope (Hz/ms)")]]
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
kmeans_vars_old = kmeans_vars
#%%
#ONLINE EXAMPLE
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
df = pd.read_csv(url, names=['sepal length','sepal width','petal length','petal width','target'])
#%%
from sklearn.preprocessing import StandardScaler
features = ['sepal length', 'sepal width', 'petal length', 'petal width']
# Separating out the features
x = df.loc[:, features].values
# Separating out the target
y = df.loc[:,['target']].values
# Standardizing the features
x = StandardScaler().fit_transform(x)
#%%
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
#%%
finalDf = pd.concat([principalDf, df[['target']]], axis = 1)
#%%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('2 component PCA', fontsize = 20)
targets = ['Iris-setosa', 'Iris-versicolor', 'Iris-virginica']
colors = ['r', 'g', 'b']
for target, color in zip(targets,colors):
    indicesToKeep = finalDf['target'] == target
    ax.scatter(finalDf.loc[indicesToKeep, 'principal component 1']
               , finalDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(targets)
ax.grid()
#%%
pca.explained_variance_ratio_
#%%
#TRYING PCA WITH MY DATA
babble_numeric = ["Delta Time (s)", "IQR BW (Hz)", "Avg Entropy (bits)", "Center Freq (Hz)", "Freq 5% Rel.", "Freq 95% Rel.", "PFC Avg Slope (Hz/ms)"]
babble_cat = ["N", "Treatment", "Nest"]
#%%
x = babble.loc[:, babble_numeric].values
y = babble.loc[:, babble_cat].values
#%%
x = StandardScaler().fit_transform(x)
#%%
pca = PCA(n_components=2)
principalComponents = pca.fit_transform(x)
principalDf = pd.DataFrame(data = principalComponents
             , columns = ['principal component 1', 'principal component 2'])
#%%
BabbleDf = pd.concat([principalDf, babble[["N", "Treatment", "Nest"]]], axis = 1)
#%%
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('7 component PCA', fontsize = 20)
treatments = ['CONTROL', 'CORT', 'OIL']
colors = ['r', 'g', 'b']
for treatment, color in zip(treatments,colors):
    indicesToKeep = BabbleDf['Treatment'] == treatment
    ax.scatter(BabbleDf.loc[indicesToKeep, 'principal component 1']
               , BabbleDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(treatments)
#%%
pca.explained_variance_ratio_
#%%
#MAKE 2D HISTOGRAMS OF NORMALIZED DATA FOR EACH TREATMENT
fig = plt.figure(figsize = (8,8))
ax = fig.add_subplot(1,1,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
ax.set_title('7 component PCA', fontsize = 20)
treatments = ['CONTROL', 'CORT', 'OIL']
colors = ['r', 'g', 'b']
for treatment, color in zip(treatments,colors):
    indicesToKeep = BabbleDf['Treatment'] == treatment
    ax.scatter(BabbleDf.loc[indicesToKeep, 'principal component 1']
               , BabbleDf.loc[indicesToKeep, 'principal component 2']
               , c = color
               , s = 50)
ax.legend(treatments)
#%%
BabbleDf_cluster = principalDf
#%%
kmeans = KMeans(n_clusters=7)
kmeans_babble = kmeans.fit_predict(BabbleDf_cluster)
#%%
labels = kmeans.predict(BabbleDf_cluster)
centroids = kmeans.cluster_centers_
#%%
LABEL_COLOR_MAP = {0: 'red', 1: 'green', 2: 'blue', 3: 'purple', 4: 'turquoise', 5: 'coral', 6: 'orange'}
label_color = [LABEL_COLOR_MAP[l] for l in kmeans_babble]
#%%
fig = plt.figure(figsize=(8, 8))

plt.scatter(BabbleDf, BabbleDf, color=label_color, alpha=0.5, edgecolor='k')
for idx, centroid in enumerate(centroids):
    plt.scatter(*centroid, color=label_color[idx+1])
plt.show()


#%%%
ax = fig.add_subplot(1,1,1) 
for treatment, color in zip(treatments,colors):
    # loop over subplots here
    indicesToKeep = BabbleDf['Treatment'] == treatment
    ax.hist2D(BabbleDf.loc[indicesToKeep, 'principal component 1']
               , BabbleDf.loc[indicesToKeep, 'principal component 2']
               )
ax.legend(treatments)

#%%
 fig = plt.figure(figsize = (12,4))
ax = fig.add_subplot(1,3,1) 
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('7 component PCA', fontsize = 20)
treatments = ['CONTROL', 'CORT', 'OIL']
colors = ['r', 'g', 'b']
treatment = treatments[0]
indicesToKeep = BabbleDf['Treatment'] == treatment
ax.hist2d(BabbleDf.loc[indicesToKeep, 'principal component 1']
           , BabbleDf.loc[indicesToKeep, 'principal component 2'])

ax = fig.add_subplot(1,3,2)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('7 component PCA', fontsize = 20)


treatment = treatments[1]
indicesToKeep = BabbleDf['Treatment'] == treatment
ax.hist2d(BabbleDf.loc[indicesToKeep, 'principal component 1']
           , BabbleDf.loc[indicesToKeep, 'principal component 2'])


ax = fig.add_subplot(1,3,3)
ax.set_xlabel('Principal Component 1', fontsize = 15)
ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('7 component PCA', fontsize = 20)


treatment = treatments[2]
indicesToKeep = BabbleDf['Treatment'] == treatment
ax.hist2d(BabbleDf.loc[indicesToKeep, 'principal component 1']
           , BabbleDf.loc[indicesToKeep, 'principal component 2'])
