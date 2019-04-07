# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
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
""" Machine Learning """
# Why we want to teach a machine a goddamn thing I'll never know.
#%%
import pandas as pd
#%%
url = "https://community.watsonanalytics.com/wp-content/uploads/2015/04/WA_Fn-UseC_-Sales-Win-Loss.csv"
#%%
""" Data Exploration """
#%% Createas a tabular data-structure
sales_data = pd.read_csv(url)
#%% Using .head() method to view the first few records of the data set
sales_data.head()
# This has to be one of the most useless things in Spyder. You can't look at a
    # fucking thing.
#%% Using .head() method with an argument which helps us to restrict the number of initial records that should be displayed
sales_data.head(n=2)
#%% Using .tail() method to view the last few records from the dataframe
sales_data.tail()
#%% Using .tail() method to view the last few records from the dataframe
sales_data.tail(n=2)
#%% Using the dtypes() method to display the different dataypes available
sales_data.dtypes
# Now THIS is useful in Spyder! Allows you to actually view all of the categories.
#%%
""" Data Visualization """
#%%
import seaborn as sns
import matplotlib.pyplot as plt
#%% Set the background color of the plot to white
sns.set(style="whitegrid", color_codes=True)

# Setting the plot size for all plots
sns.set(rc={'figure.figsize':(11.7, 8.27)})

# Create a countplot
sns.countplot('Route To Market', data=sales_data, hue='Opportunity Result')

# Remove the top and down margin
sns.despine(offset=10, trim=True)
#%%
""" Making a Violin Plot """
sns.set(rc={'figure.figsize':(10, 8)})

#Plotting the Violin Plot
sns.violinplot(x="Opportunity Result",y="Client Size By Revenue",hue="Opportunity Result", data=sales_data);
plt.show()
#%%
""" Preprocessing Data """
#%%
"""
Algorithims in scikit-learn only understant how to make changes using numeric
data, cannot use strings. Therefore, must change strings into numeric data
by arbitrarily setting numbers to things that aren't numbers because that 
really simplifies things.
"""
#%%
from sklearn import preprocessing
#%% Create the Labelencoder object
# Uses the first letter of the list to put the labelling in alphabetical order
le = preprocessing.LabelEncoder()

# Convert the categorical columns into numeric
    # .fit_transform() funtion differtiates between different unique classes
    # of the list
encoded_value = le.fit_transform(["paris", "paris", "tokyo", "amsterdam"])

# Print values

print(encoded_value)
#%%
"""
Ok, so now '1' stands for Paris, '2' stands for Tokyo, and '0' stands for
Amsterdam. Because this is simple, right?
"""
#%%
print("Supplies Subgroup' : ",sales_data['Supplies Subgroup'].unique())
print("Region : ",sales_data['Region'].unique())
print("Route To Market : ",sales_data['Route To Market'].unique())
print("Opportunity Result : ",sales_data['Opportunity Result'].unique())
print("Competitor Type : ",sales_data['Competitor Type'].unique())
print("'Supplies Group : ",sales_data['Supplies Group'].unique())
#%% Convert the categorical columns into numeric
# fit_transform() converts the categorical columns into numeric
sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])
#%%
# Display the initial records
sales_data.head()
#%%
""" Training Set & Test Set """
#%% Select columns other than 'Opportunity Number', 'Opportunity Result'
cols = [col for col in sales_data.columns if col not in ['Opportunity Number', 'Opportunity Result']]

"""
A Machine Learning algorithm needs to be trained on a set of data to learn the
relationships between different features and how these features affect the
target variable. For this we need to divide the entire data set into two sets.
One is the training set on which we are going to train our algorithm to build
a model. The other is the testing set on which we will test our model to see
how accurate its predictions are.

But before doing all this splitting, let’s first separate our features and
target variables. As before in this tutorial, we will first run the code below,
and then take a closer look at what it does:
"""

# Dropping the 'Opportunity Number' and 'Opportunity Result' columns
    # This is the FEATURE set
data = sales_data[cols]

# Assigning the Oppurtunity Result column as target
    # Creates the new TARGET dataframe
target = sales_data['Opportunity Result']

data.head(n=2)
#%%
"""
Suggests you use 80/20 (keep the 80 for training and 20 for testing)

This guy decides to use 70/30 for some arbitrary reason
"""
#%%
from sklearn.model_selection import train_test_split

#split data set into train and test sets
data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)
#%%
""" Building the Model """
#%% Choosing the Model
"""
More than 50 samples - Check
Are we predicting a category - Check
We have labeled data? (data with clear names like opportunity amount, etc.) - Check
Less than 100k samples - Check

Could be Naive Bayes
Linear SVC
K-Neightbours Classifier

Try out the different algorithms one by one.
"""
#%% Assumes every pair is independent and a Bayes theorem


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
X,y = make_moons(noise=0.3, random_state=12)
plt.scatter(X[:,0],X[:,1], c = y)
#%%
# make some data to play with (this is from sklearn.datasets, loaded above)
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

plt.scatter(X[:,0],X[:,1], c= y)
#%%
# put the three types of datasets together in one array
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]
#%%

# get the first dataset, which is moons
X, y = datasets[0]

# scale the data by the mean and standard deviation, i.e. z = (x - u) / s, 
# where x is the data, u is the mean, and s is the standard deviation
# see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X = StandardScaler().fit_transform(X)
#%%
# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
#%%
  #######
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
