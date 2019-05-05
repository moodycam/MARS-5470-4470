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
           , BabbleDf.loc[indicesToKeep0, 'principal component 2'], bins = 90, range = [[-4,6], [-6,8]], cmap = "Greys")


ax[1].set_xlabel('Principal Component 1', fontsize = 15)
ax[1].set_ylabel('Principal Component 2', fontsize = 15)
ax[1].set_title('CORT')
treatment1 = treatments[1]
indicesToKeep1 = BabbleDf['Treatment'] == treatment1
ax[1].hist2d(BabbleDf.loc[indicesToKeep1, 'principal component 1']
           , BabbleDf.loc[indicesToKeep1, 'principal component 2'], bins = 90, range = [[-4,6], [-6,8]], cmap = "Greys")

ax[2].set_xlabel('Principal Component 1', fontsize = 15)
ax[2].set_ylabel('Principal Component 2', fontsize = 15)
ax[2].set_title('OIL')
treatment2 = treatments[2]
indicesToKeep2 = BabbleDf['Treatment'] == treatment2
im = ax[2].hist2d(BabbleDf.loc[indicesToKeep2, 'principal component 1']
           , BabbleDf.loc[indicesToKeep2, 'principal component 2'], bins = 90, range = [[-4,6], [-6,8]], cmap = "Greys")

fig.colorbar(im[3], ax = ax)
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
principalComponents6 = pca.fit_transform(x)
principalDf6 = pd.DataFrame(data = principalComponents6
             , columns = ['principal component 1', 'principal component 2', 'principal component 3', 'principal component 4', 'principal component 5', 'principal component 6'])
#%%
pca.explained_variance_ratio_
#%%
pca.explained_variance_
#%%
BabbleDf6 = pd.concat([babble[["N", "Treatment", "Nest"]], principalDf6], axis = 1)
#%%
index = BabbleDf.index
columns = BabbleDf.columns
values = BabbleDf.values
#%%
controlPCA = BabbleDf[BabbleDf["Treatment"] == "CONTROL"]
cortPCA = BabbleDf[BabbleDf["Treatment"] == "CORT"]
oilPCA = BabbleDf[BabbleDf["Treatment"] == "OIL"]
#%%
fig, ax= plt.subplots(1, 3, figsize = (12,4), sharey = True)


treatments = ['CONTROL', 'CORT', 'OIL']

#make arange for bins using both PCAs

ax[0].set_xlabel('Principal Component 1', fontsize = 15)
ax[0].set_ylabel('Principal Component 2', fontsize = 15)
ax[0].set_title('CONTROL')
treatment0 = treatments[0]
indicesToKeep0 = controlPCA['Treatment'] == treatment0
ax[0].hist2d(controlPCA.loc[indicesToKeep0, 'principal component 1'], controlPCA.loc[indicesToKeep0, 'principal component 2'], bins = 90, normed = True, cmap = "Greys")


ax[1].set_xlabel('Principal Component 1', fontsize = 15)
ax[1].set_ylabel('Principal Component 2', fontsize = 15)
ax[1].set_title('CORT')
treatment1 = treatments[1]
indicesToKeep1 = cortPCA['Treatment'] == treatment1
ax[1].hist2d(cortPCA.loc[indicesToKeep1, 'principal component 1'], cortPCA.loc[indicesToKeep1, 'principal component 2'], bins = 90, normed = True, cmap = "Greys")

ax[2].set_xlabel('Principal Component 1', fontsize = 15)
ax[2].set_ylabel('Principal Component 2', fontsize = 15)
ax[2].set_title('OIL')
treatment2 = treatments[2]
indicesToKeep2 = oilPCA['Treatment'] == treatment2
im = ax[2].hist2d(oilPCA.loc[indicesToKeep2, 'principal component 1'], oilPCA.loc[indicesToKeep2, 'principal component 2'], bins = 90, normed = True, cmap = "Greys")

fig.colorbar(im[3], ax = ax)
#%%
#FIGURE OUT PCA USING STATSMODELS
from statsmodels.sandbox.tools import pca
#%%
outputbabble = pca(x, keepdim = 0, demean = True)
#%%
eigenvectors = outputbabble[3]
eigenvalues = outputbabble[2]
total_eigenvectors = outputbabble[1]
#%%
columns = ["Principal Component 1", "Principal Component 2", "Principal Component 3", "Principal Component 4", "Principal Component 5", "Principal Component 6", "Principal Component 7"]
rows = ["Delta Time (s)", "IQR BW (Hz)", "Avg Entropy (bits)", "Center Freq (Hz)", "Freq 5% Rel.", "Freq 95% Rel.", "PFC Avg Slope (Hz/ms)"]
PCs_Variables_Babble = pd.DataFrame(data=eigenvectors, index = rows, columns = columns)
PCA2_Babble = pd.DataFrame(data = total_eigenvectors, columns = columns)
#%%
BabbleDf2 = pd.concat([babble[["N", "Treatment", "Nest"]], PCA2_Babble], axis = 1)
#%%
controlPCA2 = BabbleDf2[BabbleDf2["Treatment"] == "CONTROL"]
cortPCA2 = BabbleDf2[BabbleDf2["Treatment"] == "CORT"]
oilPCA2 = BabbleDf2[BabbleDf2["Treatment"] == "OIL"]
#%%
fig, ax= plt.subplots(3, 3, figsize = (12,12), sharey = True)
fig.subplots_adjust(hspace = 0.3)

treatments = ['CONTROL', 'CORT', 'OIL']

#make arange for bins using both PCAs

ax[0,0].set_xlabel('Principal Component 1', fontsize = 10)
ax[0,0].set_ylabel('Principal Component 2', fontsize = 10)
ax[0,0].set_title('CONTROL PCA 1 and 2')
treatment0 = treatments[0]
indicesToKeep0 = controlPCA2['Treatment'] == treatment0
ax[0,0].hist2d(controlPCA2.loc[indicesToKeep0, 'Principal Component 1'], controlPCA2.loc[indicesToKeep0, 'Principal Component 2'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")


ax[0,1].set_xlabel('Principal Component 1', fontsize = 10)
ax[0,1].set_ylabel('Principal Component 2', fontsize = 10)
ax[0,1].set_title('CORT PCA 1 and 2')
treatment1 = treatments[1]
indicesToKeep1 = cortPCA2['Treatment'] == treatment1
ax[0,1].hist2d(cortPCA2.loc[indicesToKeep1, 'Principal Component 1'], cortPCA2.loc[indicesToKeep1, 'Principal Component 2'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")

ax[0,2].set_xlabel('Principal Component 1', fontsize = 10)
ax[0,2].set_ylabel('Principal Component 2', fontsize = 10)
ax[0,2].set_title('OIL PCA 1 and 2')
treatment2 = treatments[2]
indicesToKeep2 = oilPCA2['Treatment'] == treatment2
im = ax[0,2].hist2d(oilPCA2.loc[indicesToKeep2, 'Principal Component 1'], oilPCA2.loc[indicesToKeep2, 'Principal Component 2'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")

ax[1,0].set_xlabel('Principal Component 1', fontsize = 10)
ax[1,0].set_ylabel('Principal Component 3', fontsize = 10)
ax[1,0].set_title('CONTROL PCA 1 and 3')
treatment0 = treatments[0]
indicesToKeep0 = controlPCA2['Treatment'] == treatment0
ax[1,0].hist2d(controlPCA2.loc[indicesToKeep0, 'Principal Component 1'], controlPCA2.loc[indicesToKeep0, 'Principal Component 3'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")


ax[1,1].set_xlabel('Principal Component 1', fontsize = 10)
ax[1,1].set_ylabel('Principal Component 3', fontsize = 10)
ax[1,1].set_title('CORT PCA 1 and 3')
treatment1 = treatments[1]
indicesToKeep1 = cortPCA2['Treatment'] == treatment1
ax[1,1].hist2d(cortPCA2.loc[indicesToKeep1, 'Principal Component 1'], cortPCA2.loc[indicesToKeep1, 'Principal Component 3'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")

ax[1,2].set_xlabel('Principal Component 1', fontsize = 10)
ax[1,2].set_ylabel('Principal Component 3', fontsize = 10)
ax[1,2].set_title('OIL PCA 1 and 3')
treatment2 = treatments[2]
indicesToKeep2 = oilPCA2['Treatment'] == treatment2
ax[1,2].hist2d(oilPCA2.loc[indicesToKeep2, 'Principal Component 1'], oilPCA2.loc[indicesToKeep2, 'Principal Component 3'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")

ax[2,0].set_xlabel('Principal Component 2', fontsize = 10)
ax[2,0].set_ylabel('Principal Component 3', fontsize = 10)
ax[2,0].set_title('CONTROL PCA 2 and 3')
treatment0 = treatments[0]
indicesToKeep0 = controlPCA2['Treatment'] == treatment0
ax[2,0].hist2d(controlPCA2.loc[indicesToKeep0, 'Principal Component 2'], controlPCA2.loc[indicesToKeep0, 'Principal Component 3'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")


ax[2,1].set_xlabel('Principal Component 2', fontsize = 10)
ax[2,1].set_ylabel('Principal Component 3', fontsize = 10)
ax[2,1].set_title('CORT PCA 2 and 3')
treatment1 = treatments[1]
indicesToKeep1 = cortPCA2['Treatment'] == treatment1
ax[2,1].hist2d(cortPCA2.loc[indicesToKeep1, 'Principal Component 2'], cortPCA2.loc[indicesToKeep1, 'Principal Component 3'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")

ax[2,2].set_xlabel('Principal Component 2', fontsize = 10)
ax[2,2].set_ylabel('Principal Component 3', fontsize = 10)
ax[2,2].set_title('OIL PCA 2 and 3')
treatment2 = treatments[2]
indicesToKeep2 = oilPCA2['Treatment'] == treatment2
ax[2,2].hist2d(oilPCA2.loc[indicesToKeep2, 'Principal Component 2'], oilPCA2.loc[indicesToKeep2, 'Principal Component 3'], bins = 90, range = [[-6,6], [-6,8]], normed = True, cmap = "pink")


fig.colorbar(im[3], ax = ax)

fig.savefig('/Users/roryeggleston/Documents/Babble3PCA.png', bbox_inches='tight', dpi=1030)
#%%
file = '/Users/roryeggleston/Documents/BabbleOutput3Nests.xlsx'
Output3 = pd.read_excel(file)
#%%
Output3.shape
#%%
OutputIndex = Output3.index
OutputColumns = Output3.columns
OutputValues = Output3.values
#%%
ControlOutput = Output3[Output3["TREATMENT"] == "CONTROL"]
ControlAge = ControlOutput["AGE OF ONSET (dph)"]
ControlBout = ControlOutput["# OF BOUTS"]
ControlSignal = ControlOutput["# OF SIGNALS"]
CortOutput = Output3[Output3["TREATMENT"] == "CORT"]
CortAge = CortOutput["AGE OF ONSET (dph)"]
CortBout = CortOutput["# OF BOUTS"]
CortSignal = CortOutput["# OF SIGNALS"]
OilOutput = Output3[Output3["TREATMENT"] == "OIL"]
OilAge = OilOutput["AGE OF ONSET (dph)"]
OilBout = OilOutput["# OF BOUTS"]
OilSignal = OilOutput["# OF SIGNALS"]
#%%
AgeOfOnset = [ControlAge, CortAge, OilAge]
NumberBouts = [ControlBout, CortBout, OilBout]
NumberSignals = [ControlSignal, CortSignal, OilSignal]
#%%
fig, ax = plt.subplots(1, 3, figsize = (10, 4))
fig.subplots_adjust(wspace = 0.5)

ax[0].boxplot(AgeOfOnset)
ax[0].set_xlabel('Treatment', fontsize = 12)
ax[0].set_ylabel('Age of onset (dph)', fontsize = 12)
ax[0].set_title('AGE OF ONSET (dph)')

ax[1].boxplot(NumberBouts)
ax[1].set_xlabel('Treatment', fontsize = 12)
ax[1].set_ylabel('# of Bouts', fontsize = 12)
ax[1].set_title('NUMBER OF BOUTS')

ax[2].boxplot(NumberSignals)
ax[2].set_xlabel('Treatment', fontsize = 12)
ax[2].set_ylabel('# of Signals', fontsize = 12)
ax[2].set_title('NUMBER OF SIGNALS')

fig.legend(["1 = CONTROL", "2 = CORT", "3 = OIL"], loc = "best")
fig.subplots_adjust(right = 0.80)
