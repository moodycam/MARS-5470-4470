# -*- coding: utf-8 -*-
"""
Created on Mon Apr  1 13:54:02 2019

@author: Miles
"""

import numpy as np
import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import geopandas as gpd
from descartes import PolygonPatch
#%%
def load_shape_file(filepath):
    """Loads the shape file desired to mask a grid.
    Args:
        filepath: Path to *.shp file
    """
    shpfile = gpd.read_file(filepath)
    return shpfile

def select_shape(shpfile, category, name):
    """Select the submask of interest from the shapefile.
    Args:
        shpfile: (*.shp) loaded through `load_shape_file`
        category: (str) header of shape file from which to filter shape.
            (Run print(shpfile) to see options)
        name: (str) name of shape relative to category.
        plot: (optional bool) if True, plot the polygon that will be masking.
    Returns:
        shapely polygon
    Example:
        from esmask.mask import load_shape_file, select_shape
        LME = load_shape_file('LMEs.shp')
        CalCS = select_shape(LME, 'LME_NAME', 'California Current')
    """
    s = shpfile
    polygon = s[s[category] == name]
    polygon = polygon.geometry[:].unary_union
    return polygon
#%%
shp = load_shape_file('C:\\Users\Miles\Downloads\LME66\LMEs66.shp')
shp.head() # .head gives you the first few lines, aka shapes
#%%
CalCS_shp = select_shape(shp, 'LME_NAME', 'California Current')
CalCS_shp
#%%
def lat_lon_formatter(ax):
    """
    Creates nice latitude/longitude labels
    for maps
    """
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=16)
    

def set_up_map(ax, x0, x1, y0, y1):
    """
    Adds coastline, etc.
    
    x0, x1: longitude bounds to zoom into
    y0, y1: latitude bounds to zoom into
    """
    # set up land overlay
    ax.add_feature(cfeature.LAND, facecolor='k')
    
    # zoom in on region of interest
    ax.set_extent([x0, x1, y0, y1])
    
    # set nicer looking ticks
    ax.set_xticks(np.arange(x0, x1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(y0, y1, 10), crs=ccrs.PlateCarree())
    lat_lon_formatter(ax)
#%%
f, ax = plt.subplots(ncols=2, figsize=(10,5),
                     subplot_kw=dict(projection=ccrs.PlateCarree()))
set_up_map(ax[0], -140, -107, 20, 50)
set_up_map(ax[1], -140, -107, 20, 50)



# add shapefile to map
ax[0].add_patch(PolygonPatch(CalCS_shp, fc='#add8e6', zorder=4))
# some other attributes to play around with
ax[1].add_patch(PolygonPatch(CalCS_shp, fc='None', ec='r', linewidth=2,
                             linestyle=':'))
#%%
import matplotlib
import cartopy.crs as ccrs
#%%
shp = load_shape_file('C:\\Users\Miles\Downloads\LME66\LMEs66.shp')
shp.head() # .head gives you the first few lines, aka shapes
#%%
# Define the CartoPY CRS object.
crs = ccrs.Orthographic()

# This can be converted into a 'proj4' string/dict compatible with GeoPandas
crs_proj4 = crs.proj4_init
df = shp.to_crs(crs_proj4)

# Here's what the plot looks like in GeoPandas
ax = df.plot(column='Shape_Area', cmap='Greens')
#%%
# The data of interest should map to the same indices as your shape files,
# such as LMEs.
total_area = shp['Shape_Area'].values
total_area
#%%
""" SUM_GIS_KM Section """
#%%
sum_gis = shp['SUM_GIS_KM'].values
#%%
def make_colormap(data, cmap_name, norm='linear'):
    """
    Make colormap for mapping shapefiles to.
    
    data: an array of data that the colormap will map to.
    cmap_name: string of a matplotlib colormap.
    norm: defaults to 'linear'. Pass 'log' for logarithmic colorbar.
    """
    cmap = matplotlib.cm.get_cmap(cmap_name)
    if norm == 'linear':
        norm = matplotlib.colors.Normalize(min(data), max(data))
    elif norm == 'log':
        norm = matplotlib.colors.LogNorm(min(data), max(data))
    else:
        raise ValueError("Supply 'linear' or 'log' for norm.")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = [] # some weird bug to make sure we can plot our colorbar later.
    return sm
#%%
f, axes = plt.subplots(figsize=(16,5),
                       subplot_kw=dict(projection=ccrs.PlateCarree()))

# Learning how to iterate cleanly is a good skill in python.
# Here, enumerate returns a tuple of values. First a counter
# (just like if you did i += 1 at the end of every loop) and second,
# the value fhmom the list you're iterating through.
#
# Here, I am just iterating through the LME names to plot them
# one by one. The i will be used to reference our total_area array
# and color it.

# Use the data we want to plot to make the colormap
sm = make_colormap(sum_gis, 'magma')


for i, LME in enumerate(shp['LME_NAME'].values):
   # Get polygon for each LME name
    poly = select_shape(shp, 'LME_NAME', LME)
    # gets the color for the corresponding area for that LME
    color = sm.to_rgba(sum_gis[i])
    # Adds the polygon patch project
    test = axes.add_patch(PolygonPatch(poly, fc=color, zorder=4))
axes.axis('scaled')
axes.add_feature(cfeature.LAND, color='k')

# add colorbar.
# reference: https://stackoverflow.com/questions/35873209/matplotlib-add-colorbar-to-cartopy-image
cb = plt.colorbar(sm, orientation='horizontal', pad=0.05, fraction=0.09)
cb.set_label('LME Area')
#%%
""" LME Name """
#%%
lme_num = shp['LME_NUMBER'].values
#%%
def make_colormap(data, cmap_name, norm='linear'):
    """
    Make colormap for mapping shapefiles to.
    
    data: an array of data that the colormap will map to.
    cmap_name: string of a matplotlib colormap.
    norm: defaults to 'linear'. Pass 'log' for logarithmic colorbar.
    """
    cmap = matplotlib.cm.get_cmap(cmap_name)
    if norm == 'linear':
        norm = matplotlib.colors.Normalize(min(data), max(data))
    elif norm == 'log':
        norm = matplotlib.colors.LogNorm(min(data), max(data))
    else:
        raise ValueError("Supply 'linear' or 'log' for norm.")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = [] # some weird bug to make sure we can plot our colorbar later.
    return sm
#%%
f, axes = plt.subplots(figsize=(16,5),
                       subplot_kw=dict(projection=ccrs.PlateCarree()))

# Learning how to iterate cleanly is a good skill in python.
# Here, enumerate returns a tuple of values. First a counter
# (just like if you did i += 1 at the end of every loop) and second,
# the value fhmom the list you're iterating through.
#
# Here, I am just iterating through the LME names to plot them
# one by one. The i will be used to reference our total_area array
# and color it.

# Use the data we want to plot to make the colormap
sm = make_colormap(lme_num, 'magma')


for i, LME in enumerate(shp['LME_NAME'].values):
   # Get polygon for each LME name
    poly = select_shape(shp, 'LME_NAME', LME)
    # gets the color for the corresponding area for that LME
    color = sm.to_rgba(lme_num[i])
    # Adds the polygon patch project
    test = axes.add_patch(PolygonPatch(poly, fc=color, zorder=4))
axes.axis('scaled')
axes.add_feature(cfeature.LAND, color='k')

# add colorbar.
# reference: https://stackoverflow.com/questions/35873209/matplotlib-add-colorbar-to-cartopy-image
cb = plt.colorbar(sm, orientation='horizontal', pad=0.05, fraction=0.09)
cb.set_label('LME Area')
#%%
""" 12.2 Work """
#%%
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# data and pre-processing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

# All the classifiers we will use:
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
#%%
"""
Color classifiers guesses which color the dot will be based on the distribution
of the data. For example; it gauges that if a dot is in the red field it is
more likely to be red.

Some have a high accuracy but appear very inaccurate. Not sure how "accuracy"
is measured.
"""
#%%
h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
#%%
# create a function list of all the classifiers
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=2, C=1),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    MLPClassifier(alpha=1),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]
#%%
# make some data to play with (this is from sklearn.datasets, loaded above)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

plt.scatter(X[:,0],X[:,1], c= y)
#%%
# add som noise to these so they smoosh together
X_old = np.copy(X)
# make a random number object
rng = np.random.RandomState(2)
# add some random numbers to the data
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
#%%
plt.scatter(X_old[:,0],X_old[:,1], c = y)
plt.colorbar()
#%%
plt.scatter(X[:,0],X[:,1], c = y)
#%%
"""
Exercise 1:
    What happens if you use X_old = X instead of X_old = np.copy(X)?
Answer:
    Just smooshes them right together and changes the pattern.
"""
#%%
# make some data to play with (this is from sklearn.datasets, loaded above)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

plt.scatter(X[:,0],X[:,1], c= y)
#%%
# add som noise to these so they smoosh together
X_old = X
# make a random number object
rng = np.random.RandomState(2)
# add some random numbers to the data
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
#%%
plt.scatter(X_old[:,0],X_old[:,1], c = y)
plt.colorbar()
#%%
plt.scatter(X[:,0],X[:,1], c = y)
#%%
""" End of Exercise 1 """
#%%
""" Moons"""
#%%
X,y = make_moons(noise=0.3, random_state=12)
plt.scatter(X[:,0],X[:,1], c = y)
#%%
"""
Exercise 2:
    What happens to the make_moons data if you change the noise and random
    state? Why do you think they picked noise = 0.3?
Answer:
    Increasing the noise and random state integrates the two groups more, while
    reducing the noise and random state separate and clarify the groups.
"""
#%%
X,y = make_moons(noise=0.1, random_state=10)
plt.scatter(X[:,0],X[:,1], c = y)
#%%
""" End of Exercise 2 """
#%%
""" Circles """
#%%
# make some data to play with (this is from sklearn.datasets, loaded above)
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

plt.scatter(X[:,0],X[:,1], c= y)
#%%
"""
Exercise 2 (Second #2):
     Play around with the options for make_circles. What does factor do?
Answer:
    Factor pushes the inner circle around. Lower factors make the circle more 
    compact, while higher factors makes the circle wider, pushing it farther
    into the circle of the other data..
"""
#%%
# make some data to play with (this is from sklearn.datasets, loaded above)
X, y = make_circles(noise=0.2, factor=0.9, random_state=1)

plt.scatter(X[:,0],X[:,1], c= y)
#%%
""" End of Exercise 2 (Second #2) """
#%%
# put the three types of datasets together in one array
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]
#%%
""" Part 2: Big Double 'for' Loop """
#%%
# get the first dataset, which is moons
X, y = datasets[0]

# scale the data by the mean and standard deviation, i.e. z = (x - u) / s, 
# where x is the data, u is the mean, and s is the standard deviation
# see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X1 = StandardScaler().fit_transform(X)
#%%
"""
Exercise 3:
    Make a plot with two subplots, showing the scaled and unscaled data next
    to each other. How are they different?
Answer:
    It changes the centroid of the thing.
"""
#%%
fig, ax = plt.subplots(figsize=(12,8), nrows=1, ncols=2)
ax[0].scatter(X[:,0],X[:,1], c=y)
ax[1].scatter(X1[:,0],X1[:,1], c=y)
#%%
""" End of Exercise 3 """
#%%
# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
#%%
"""
Exercise 4:
    What percentage of the data is used for training here?
Answer:
    40%.
"""
#%%
""" Plot the Data """
# set up a meshgrid for plotting the classification result based on the size of the dataset
# note this will be used later
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
#######

# counter for which subplot we are in
i = 1

# this counter is used to know when to plot the title
ds_cnt = 0

# just plot the dataset first

# note this is not used yet
cm = plt.cm.RdBu

# set the colormap used for the data 
# see https://matplotlib.org/tutorials/colors/colorbar_only.html#sphx-glr-tutorials-colors-colorbar-only-py
cm_bright = ListedColormap(['#FF0000', '#0000FF'])

ax = plt.subplot(1,1, i)
#ax = plt.subplot(len(datasets), len(classifiers) + 1, i)

# if the first row, plot the title
if ds_cnt == 0:
    ax.set_title("Input data")
    
# Plot the training points using the bright colormap
ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
           edgecolors='k')

# set the limits to the colormap min max we will use later
ax.set_xlim(xx.min(), xx.max())
ax.set_ylim(yy.min(), yy.max())

# get rid of ticks
ax.set_xticks(())
ax.set_yticks(())

# increment to go to the next subplot
i += 1
#%%
""" Nearest Neighbor """
"""
Looks at 'k' # of nearest neighbors, using some distance metric, and predicts
which class a point is in based on the type of thing those neighbors are
"""
#%%
# get the first classifier, KNeighborsClassifier(3)
# here we are using 3 nearest neighbors

# index for classifier
c = 0

name = names[0]
clf = classifiers[0]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
#%%
name
#%%
clf

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance
#%%
score
#%%
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#%%
"""
Exercise 5:
    What does the xx.ravel() bit of code do?
Answer:
    It creates a 1-D array containing the elements of the input. The returned
    array has the same type as the input array (i.e. masked remains masked)
"""
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()
#%%
"""
Exercise 6:
    What do the light blue and peach colors represent?
Answer:
    Those are areas where the dark red and blue meet, meaning the machine 
    learning cannot easily predict whether a point would definitely be red or
    blue. For example, if a point is in dark blue there is a very high chance
    that the point would be blue. If it is in light blue there is a lower
    chance it is definitely blue.
"""
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)
#%%
"""
Exercise 7:
    Q: How well did the method do?
        A: It appears to have done very well. Only one red point is within the
        dark blue area and no dark blue dots are within any of the dark red
        areas.
    Q: Are their outliers in either class?
        A: Yes, however there are more red outliers than blue outilers. The 
        majority of the blue points remain clumped together, where the red
        points are a bit more spread out.
    Q: Is it reasonable that this method did not predict those, given the training dataset?
        Yes. None of the red points are isolated away from blue points, they are
        all relatively close to blue points, clouding the data where they meet.
"""
#%%
"""
Exercise 8:
    Using one of the other data sets and methods, go through the same exercise
    as above. Explain the basics of what your method does, and how well it
    works for the dataset.
Answer:
    Chose to use RBF SVM (SVC in 'classifiers' list)
    It seems to work well for the dataset. It states its accuracy score is
    0.875, which seems a little high for how true it is to the data. This
    method also seems incredibly conservative. The sections it claims are very
    likely are small compared to the areas it states are only somewhat likely.
"""
#%%
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
#%%
# index for classifier
c = 2

name = names[2]
clf = classifiers[2]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
#%%
name
#%%
clf

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance
#%%
score
#%%
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)
#%%
"""
Exercise 9:
    Again, using one of the other data sets and methods, go through the same
    exercise as above. Explain the basics of what your method does, and how
    well it works for the dataset.
Answer:
    These groups remain rather separated, using the linear dataset and 
    Gaussian NB classifier. It claimed a 1.0 score, which is correct because
    the groups were created so far apart.
"""
#%%
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
#%%
# index for classifier
c = 8

name = names[8]
clf = classifiers[8]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
#%%
name
#%%
clf

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance
#%%
score
#%%
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)
#%%
"""
Exercises 10-11:
    Pick a few more to go through.
"""
#%%
"""
Exercise 10:
"""
"""
Answer:
    This time tried the 'Random Forest' classifier, which has a score of 0.75.
"""
#%%
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
# split the data into training and testing
# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
#%%
# index for classifier
c = 5

name = names[5]
clf = classifiers[5]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
#%%
name
#%%
clf

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance
#%%
score
#%%
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)
#%%
"""
Exercise 11:
"""
"""
Answer:
    This time tried the 'Gaussian Process' classifier, which has a score of
    0.9.
"""
#%%
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
# split the data into training and testing
# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
#%%
# index for classifier
c = 3

name = names[3]
clf = classifiers[3]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)
#%%
name
#%%
clf

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance
#%%
# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].
if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()
#%%
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)