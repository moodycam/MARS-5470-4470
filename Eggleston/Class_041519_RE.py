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
fig, ax= plt.subplots(1, 3, figsize = (12,4), sharey = True)

treatments = ['CONTROL', 'CORT', 'OIL']

#make arange for bins using both PCAs

ax[0].set_xlabel('Principal Component 1', fontsize = 15)
ax[0].set_ylabel('Principal Component 2', fontsize = 15)
ax[0].set_title('CONTROL')
treatment0 = treatments[0]
indicesToKeep0 = BabbleDf['Treatment'] == treatment0
ax[0].hist2d(BabbleDf.loc[indicesToKeep0, 'principal component 1']
           , BabbleDf.loc[indicesToKeep0, 'principal component 2'], bins = 90)



ax[1].set_xlabel('Principal Component 1', fontsize = 15)
ax[1].set_ylabel('Principal Component 2', fontsize = 15)
ax[1].set_title('CORT')
treatment1 = treatments[1]
indicesToKeep1 = BabbleDf['Treatment'] == treatment1
ax[1].hist2d(BabbleDf.loc[indicesToKeep1, 'principal component 1']
           , BabbleDf.loc[indicesToKeep1, 'principal component 2'], bins = 90)


ax[2].set_xlabel('Principal Component 1', fontsize = 15)
ax[2].set_ylabel('Principal Component 2', fontsize = 15)
ax[2].set_title('OIL')
treatment2 = treatments[2]
indicesToKeep2 = BabbleDf['Treatment'] == treatment2
ax[2].hist2d(BabbleDf.loc[indicesToKeep2, 'principal component 1']
           , BabbleDf.loc[indicesToKeep2, 'principal component 2'], bins = 90)
#%%
index = kmeans_vars.index
columns = kmeans_vars.columns
values = kmeans_vars.values
#%%
control = kmeans_vars[kmeans_vars["Treatment"] == "CONTROL"]
cort = kmeans_vars[kmeans_vars["Treatment"] == "CORT"]
oil = kmeans_vars[kmeans_vars["Treatment"] == "OIL"]
#%%
fig, ax = plt.subplots(nrows = 3, ncols = 7, figsize = (12,4), sharey = True)
plt.rc('xtick',labelsize=6)
plt.rc('ytick',labelsize=6)
plt.suptitle("Distribution of Babbling Variables")

ax[0,0].set_ylabel("CONTROL", rotation=90, size='small')
ax[0,0].hist(control["IQR BW (Hz)"])
ax[0,0].set_title("IQR BW (Hz)", fontsize = 9)
ax[0,1].hist(control["Delta Time (s)"])
ax[0,1].set_title("Delta Time (s)", fontsize = 9)
ax[0,2].hist(control["Avg Entropy (bits)"])
ax[0,2].set_title("Avg Entropy (bits)", fontsize = 9)
ax[0,3].hist(control["Center Freq (Hz)"])
ax[0,3].set_title("Center Freq (Hz)", fontsize = 9)
ax[0,4].hist(control["Freq 5% Rel."])
ax[0,4].set_title("Freq 5% Rel.", fontsize = 9)
ax[0,5].hist(control["Freq 95% Rel."])
ax[0,5].set_title("Freq 95% Rel.", fontsize = 9)
ax[0,6].hist(control["PFC Avg Slope (Hz/ms)"])
ax[0,6].set_title("PFC Avg Slope (Hz/ms)", fontsize = 9)

ax[1,0].set_ylabel("CORT", rotation=90, size='small')
ax[1,0].hist(cort["IQR BW (Hz)"])
ax[1,1].hist(cort["Delta Time (s)"])
ax[1,2].hist(cort["Avg Entropy (bits)"])
ax[1,3].hist(cort["Center Freq (Hz)"])
ax[1,4].hist(cort["Freq 5% Rel."])
ax[1,5].hist(cort["Freq 95% Rel."])
ax[1,6].hist(cort["PFC Avg Slope (Hz/ms)"])

ax[2,0].set_ylabel("OIL", rotation=90, size='small')
ax[2,0].hist(oil["IQR BW (Hz)"])
ax[2,1].hist(oil["Delta Time (s)"])
ax[2,2].hist(oil["Avg Entropy (bits)"])
ax[2,3].hist(oil["Center Freq (Hz)"])
ax[2,4].hist(oil["Freq 5% Rel."])
ax[2,5].hist(oil["Freq 95% Rel."])
ax[2,6].hist(oil["PFC Avg Slope (Hz/ms)"])
#%%
#fit determines mean and SD
#transform applies them
#fit_transform will give the same as just fit if you've already used StandardScaler on the data
pca = PCA(n_components=6)
principalComponents3 = pca.fit_transform(x)
principalDf3 = pd.DataFrame(data = principalComponents3
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6'])
#%%
pca.explained_variance_ratio_
#%%

