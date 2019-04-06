# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 16:05:32 2019

@author: tomyamashita
"""

# Week 11 Code stuff

#%% 4/1/2019

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
%matplotlib inline
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import geopandas as gpd
from descartes import PolygonPatch

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
    ax.set_xticks(np.arange(x0, x1, 30), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(y0, y1, 30), crs=ccrs.PlateCarree())
    lat_lon_formatter(ax)
    
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
        norm = matplotlib.colors.LogNorm(min(data), max(data))  # This used to be 'Total_area' instead of data
    else:
        raise ValueError("Supply 'linear' or 'log' for norm.")
    sm = matplotlib.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm._A = [] # some weird bug to make sure we can plot our colorbar later.
    return sm



#%% Warmup Exercise 1
    
shp = load_shape_file('LMEs66.shp')
shp.head()

USLME = shp[shp['USLMES'] == 'Yes']

KM2 = USLME['SUM_GIS_KM'].values
KM2


f, ax = plt.subplots(figsize=(10,5), subplot_kw=dict(projection=ccrs.PlateCarree()))
   
sm = make_colormap(KM2, 'Blues')
for i, LME in enumerate(USLME['LME_NAME'].values):
    poly = select_shape(USLME, 'LME_NAME', LME)
    color = sm.to_rgba(KM2[i])
    test = ax.add_patch(PolygonPatch(poly, fc=color, zorder = 4))

set_up_map(ax, -180, -45, 15, 85)

#ax.axis('scaled')  This is not a label

ax.add_feature(cfeature.LAND, color = 'g')

cb = plt.colorbar(sm, orientation = 'horizontal', pad = 0.15, fraction = 0.09)
cb.set_label('LME Area (km^2)')

#%% Modifying the map. Warmup Exercise 2

f, ax = plt.subplots(figsize=(10,5), subplot_kw=dict(projection=ccrs.PlateCarree()))
   
sm = make_colormap(KM2, 'Reds', norm = 'log')
for i, LME in enumerate(USLME['LME_NAME'].values):
    poly = select_shape(USLME, 'LME_NAME', LME)
    color = sm.to_rgba(KM2[i])
    test = ax.add_patch(PolygonPatch(poly, fc=color, zorder = 4))

set_up_map(ax, -180, -45, 15, 85)

#ax.axis('scaled')  This is not a label

ax.add_feature(cfeature.LAND, color = 'g')

cb = plt.colorbar(sm, orientation = 'vertical', pad = 0.05, fraction = 0.09)
cb.set_label('LME Area (km^2)')

#%% Lecture 11.1

import pandas as pd

url = "https://community.watsonanalytics.com/wp-content/uploads/2015/04/WA_Fn-UseC_-Sales-Win-Loss.csv"

#%% Read the dataset

sales_data = pd.read_csv(url)

#%% Looking at the data
print(sales_data.head())
print(sales_data.tail())
sales_data.dtypes

#%% Visualizing data

import seaborn as sns
import matplotlib.pyplot as plt

sns.set(style = 'whitegrid', color_codes = True)

sns.set(rc = {'figure.figsize':(11.7, 8.27)})

sns.countplot('Route To Market', data = sales_data, hue = 'Opportunity Result')

sns.despine(offset = 10, trim = True)

#%% Violin Plot

# seaborn's version of setting the figure size
sns.set(rc={'figure.figsize':(16.7, 13.27)})

sns.violinplot(x = 'Opportunity Result', y = 'Client Size By Revenue', hue = 'Opportunity Result', data = sales_data)
plt.show()

#%% Preprocessing data

# Convert text data into coded values

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

encoded_value = le.fit_transform(['paris', 'paris', 'tokyo', 'amsterdam'])

# This method encodes data based on alphabetical order

print(encoded_value)

#%% Encode categorical data from sales data

print("Supplies Subgroup' : ",sales_data['Supplies Subgroup'].unique())
print("Region : ",sales_data['Region'].unique())
print("Route To Market : ",sales_data['Route To Market'].unique())
print("Opportunity Result : ",sales_data['Opportunity Result'].unique())
print("Competitor Type : ",sales_data['Competitor Type'].unique())
print("'Supplies Group : ",sales_data['Supplies Group'].unique())

#%% Convert categorical data

sales_data['Supplies Subgroup'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group'] = le.fit_transform(sales_data['Supplies Group'])

sales_data.head()

#%% Creating a training and testing set

cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]

data = sales_data[cols]

target = sales_data['Opportunity Result']

data.head()

#%% Import module and create training samples

from sklearn.model_selection import train_test_split

data_train, data_test, target_train, target_test = train_test_split(data,target, test_size = 0.30, random_state = 10)

#%% Gaussian Naive Bayes algorith

from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import accuracy_score

gnb = GaussianNB()

# Create a training sample and predict it using the test sample
pred = gnb.fit(data_train, target_train).predict(data_test)
#print(pred.tolist())

# Check the accuracy of the test sample
print("Naive-Bayes accuracy : ", accuracy_score(target_test, pred, normalize = True))

#%% LinearSVC

from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score

# Create a LinearSVC object
svc_model = LinearSVC(random_state = 0)

# Train algorithm on training data and test
pred = svc_model.fit(data_train, target_train).predict(data_test)

print('LinearSVC accuracy : ', accuracy_score(target_test, pred, normalize = True))

#%% K-Neighbor Classifier

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Create object for the classifier
neigh = KNeighborsClassifier(n_neighbors = 3)

# Train the algorithm
neigh.fit(data_train, target_train)

# Predict the response
pred = neigh.predict(data_test)

print("KNeighbors accuracy score : ", accuracy_score(target_test, pred))

#%% Performance comparison (This package doesn't exist)

"""
from yellowbrick.classifier import ClassificationReport

visualizer = ClassificationReport(gnb, classes = ['Won', 'Loss'])

visualizer.fit(data_train, target_train)
visualizer.score(data_test, target_test)
g = visualizer.poof()
"""

#%% Lecture Exercises: Exercise 1

sns.despine?

"""
The function supposedly removes the spines from the top and right of the plot
However, when you run the same code without the despine function, nothing changes making it not very useful
"""

#%% Exercise 2

sns.set(rc={'figure.figsize':(16.7, 13.27)})

sns.set(font_scale = 2) # This increases the size of the text in the plot by double
sns.violinplot(x = 'Opportunity Result', y = 'Client Size By Revenue', hue = 'Opportunity Result', data = sales_data)
plt.show()

#%% Exercise 3

from sklearn import preprocessing

le = preprocessing.LabelEncoder()

encoded_value = le.fit_transform(['paris', 'paris', 'tokyo', 'amsterdam'])
print(encoded_value)

modified_encoder = le.fit_transform(['paris', 'Paris', 'tokyo', 'amsterdam'])
print(modified_encoder)

"""
Yes, the label encoder is case sensitive. Values with start with capital letters go first then, lower case letters
"""

#%% Exercise 4

sales_data['Supplies Subgroup1'] = le.fit_transform(sales_data['Supplies Subgroup'])
sales_data['Region1'] = le.fit_transform(sales_data['Region'])
sales_data['Route To Market1'] = le.fit_transform(sales_data['Route To Market'])
sales_data['Opportunity Result1'] = le.fit_transform(sales_data['Opportunity Result'])
sales_data['Competitor Type1'] = le.fit_transform(sales_data['Competitor Type'])
sales_data['Supplies Group1'] = le.fit_transform(sales_data['Supplies Group'])

sales_data.head()

"""
You can create new columns by using a unique name for the definition of the encoded data
"""

#%% Exercise 5

cols = [col for col in sales_data.columns if col not in ['Opportunity Number','Opportunity Result']]

col 
for col in sales_data.columns:
    if col not in ['Opportunity Number','Opportunity Result']]

"""
The code looks up the column names in the dataset
Then it includes all columns that are not Opportunity Number or Opportunity Result
And outputs a list with the remaining columns
"""

#%%Exercise 6

train_test_split?

"""
Train test split takes a number of arrays and splits them into training samples and testing samples and outputs them as their own objects
Random_state determines the randomization method used to pull from the dataset
"""

#%% Exercise 7

data.count()
data_test.count()
data_train.count()

"""
The test size is of size 23408 and the training size is 54617
The test size is 30% of the total data which is 78025 

Using the test_size object within the train_test_split function sets the percentage of data points to be used as the test size
The training size is the rest
To have 80% of data as training sample, set test_size = 0.2 instead of 0.3
"""

#%% Exercise 8

"""
The Naive-Bayes accuracy is the probability that any individual point that was interpolated from the test data is correct
Higher values are those that have a higher chance of being right

Given a particular consumer who encompasses certain characteristics (the other variables), one can predict the probability of making a sale towards that consumer
I.e. the probality that you will succeed given a set of conditions
"""

#%% 4/3/2019 Warmup Exercise

def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    
    return u + dt * f(u)

def RK4(u,f,dt):
    # Runge Kutta 4th order method
    """Returns the solution at the next time-step using Runge Kutta fourth order (RK4) method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    #calculate slopes
    k1 = f(u)
    u1 = u + (dt/2.)*k1
    k2 = f(u1)
    u2 = u + (dt/2.)*k2
    k3 = f(u2)
    u3 = u + dt*k3
    k4 = f(u3)
    return u + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
   
def f(u):
    """Returns the rate of change of species numbers.
    
    Parameters
    ----------
    u : array of float
        array containing the solution at time n.
        
    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """
    x = u[0]
    y = u[1]
    return np.array([x*(alpha - beta*y), -y*(gamma - delta*x + r*y)])
 
import numpy as np
import matplotlib.pyplot as plt

alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
r = gamma/10
T = 15.0            # Final time
dt = 0.01           # Time increment
N = int(T/dt) + 1   # Number of time steps = Final time / time increment + 1 as an integer
x0 = 10             # Initial x value (prey?)
y0 = 2              # Initial y value (predator?)
t0 = 5   

#%% 

u_euler = np.empty((N,2))  
u_euler[0] = np.array([x0,y0])
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
    
time = np.linspace(t0, T+t0, N)  
x_euler = u_euler[:, 0]
y_euler = u_euler[:, 1]

plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Predator prey model')

#%% System behavior

plt.plot(x_euler, y_euler, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')

#%% RK method

u_RK = np.empty((N,2))  
u_RK[0] = np.array([x0,y0])
for n in range(N-1):
    u_RK[n+1] = RK4(u_RK[n], f, dt)
    
time = np.linspace(t0, T+t0, N)  
x_RK = u_RK[:, 0]
y_RK = u_RK[:, 1]

plt.plot(time, x_RK, label = 'Prey')
plt.plot(time, y_RK, label = 'Predator')
plt.legend(loc='upper left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Predator prey model')

#%% System behavior

plt.plot(x_RK, y_RK, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')


#%% Lecture 11.2

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

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

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

figure = plt.figure(figsize=(27, 9), facecolor = 'w')
i = 1
# iterate over datasets
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    X, y = ds
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='green', alpha=1, zorder = 10)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            ax.set_title(name, fontsize = 14)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1

plt.tight_layout()

#%% Breaking it down

# Loading a bunch of packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Data and preprocessing packages
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification

# Classifiers
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

h = .02  # step size in the mesh. Density of the points

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

#%% Make list of classifiers

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

#%% make some data to play with (this is from sklearn.datasets, loaded above)
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)

plt.scatter(X[:,0],X[:,1], c= y) # Visualize the clusters

#%% Add som noise to these so they smoosh together

X_old = np.copy(X)
# make a random number object
rng = np.random.RandomState(2)
# add some random numbers to the data
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

plt.scatter(X_old[:,0],X_old[:,1], c = y)
plt.colorbar()

#%% 
plt.scatter(X[:,0],X[:,1], c= y)

#%% Make moons

X,y = make_moons(noise=0.3, random_state=12)
plt.scatter(X[:,0],X[:,1], c = y)

#%% Make circular data

# make some data to play with (this is from sklearn.datasets, loaded above)

X, y = make_circles(noise=0.2, factor=0.5, random_state=1)

plt.scatter(X[:,0],X[:,1], c= y)

#%% put the three types of datasets together in one array

datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

#%% Big four loop

# get the first dataset, which is moons
X, y = datasets[0]

# scale the data by the mean and standard deviation, i.e. z = (x - u) / s, 
# where x is the data, u is the mean, and s is the standard deviation
# see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X = StandardScaler().fit_transform(X)

# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
    
#%% Plotting the Data
    
# set up a meshgrid for plotting the classification result based on the size of the dataset
# note this will be used later
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


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

#%% Nearest Neighbors

# get the first classifier, KNeighborsClassifier(3)
# here we are using 3 nearest neighbors

# index for classifier
c = 0

name = names[0]
clf = classifiers[0]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(name)

print(clf)

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance

print(score)

#%% Plot things

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the testing and training points on top of the decision matrix mesh

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)

#%% Exercise 1

X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
plt.scatter(X[:,0],X[:,1], c= y) # Visualize the clusters


X_old = X
# make a random number object
rng = np.random.RandomState(2)
# add some random numbers to the data
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

plt.scatter(X_old[:,0],X_old[:,1], c = y)
plt.colorbar()

X==X_old

"""
np.copy creates a new dataset that is identical to X so if you modify one, it doesn't modify the other. 
If you don't use np.copy, if you make modifications, it modifies the original dataset too
"""

#%% Exercise 2

X,y = make_moons(noise=0.1, random_state=3)
plt.scatter(X[:,0],X[:,1], c = y)

"""
There is greater overlap between 0's and 1's with greater noise. Each dataset is more dispersed
They probably used 0.3 because the 2 datasets are still distinct but almost overlap
Changing the random state just changes what order the points will be chosen in
"""

#%% Exercise 2 (again)

X, y = make_circles(noise=0.3, factor=0.4, random_state=1)
plt.scatter(X[:,0],X[:,1], c= y)

"""
Factor is a scaling factor between the inner and outer circles so changing the factor changes how much overlap there is between circles
"""

#%% Exercise 3

# get the first dataset, which is moons
X, y = datasets[0]

# Create copy of X before scaling the data
X_unscaled = np.copy(X)

# scale the data by the mean and standard deviation, i.e. z = (x - u) / s, 
# where x is the data, u is the mean, and s is the standard deviation
# see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X = StandardScaler().fit_transform(X)


fig, ax = plt.subplots(ncols = 2, figsize = (8,4))
ax[0].scatter(X[:,0], X[:,1], c = y)
ax[1].scatter(X_unscaled[:,0], X_unscaled[:,1], c = y)

"""
The only difference between plots is the scales of the x and y axes
The shape of the data is the same
"""

#%% Exercise 4

# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
    
"""
The training data percent is 60% because 40% is used for the testing data
"""

#%% Exercise 5

print(xx.shape)
print(xx.ravel().shape)

"""
The ravel function turns a multidimensional array into a one-dimensional array
of size equal to the total number of cells in the array
"""

#%% Exercise 6

# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()

"""
The lighter colors in the plot represent areas where the probability of finding a blue or a red point are lower than 1
Light blue areas are those that are most likely blue but the algorithm was not sure that it is blue for sure
"""

#%% Exercise 7

# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the testing and training points on top of the decision matrix mesh

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)

"""
The algorithm did pretty good. 
Only one testing point (a red) was in the area where blue is supposed to be
For the most part, where there was overlapping blue and red training points or where there were no points, the probability was lower for a blue or red 
"""

#%% Exercise 8

# Redefine X and y because I don't know what is being changed each time
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)
X,y = make_moons(noise=0.3, random_state=12)
X, y = make_circles(noise=0.2, factor=0.5, random_state=1)
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

# Create copy so can easily reuse to recreate original X and y if necessary
X_copy, y_copy = np.copy(X), np.copy(y)

#%% Choose a dataset
## 0 = moons
## 1 = circles
## 2 = linear

X, y = datasets[1] # Choose the circles dataset

# scale the data by the mean and standard deviation, i.e. z = (x - u) / s, 
# where x is the data, u is the mean, and s is the standard deviation
# see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X = StandardScaler().fit_transform(X)

# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)
    
#%% Plotting the Data

# This is only necessary the first time and is not necessary for each of the future iterations of this

    
# set up a meshgrid for plotting the classification result based on the size of the dataset
# note this will be used later
x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))


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

#%% Linear SVM

# index for classifier
c = 1  # Choose the Linear SVM classifier

name = names[c] # Use c for the classifier method to allow for modifying of method easily
clf = classifiers[c]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(name)

print(clf)

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance

print(score)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the testing and training points on top of the decision matrix mesh

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)

"""
The Linear SVM method is quite a poor predictor for the circular data
It seems to use a linear approximation for the data so for a circular dataset, it works very poorly
"""

#%% Exercise 9. RBF SVM

# Reset X and y
X, y = np.copy(X_copy), np.copy(y_copy)

X, y = datasets[2] # Choose the linear dataset

# scale the data by the mean and standard deviation, i.e. z = (x - u) / s, 
# where x is the data, u is the mean, and s is the standard deviation
# see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X = StandardScaler().fit_transform(X)

# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

# index for classifier
c = 2  # Choose the RBF SVM classifier

name = names[c] # Use c for the classifier method to allow for modifying of method easily
clf = classifiers[c]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(name)

print(clf)

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance

print(score)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the testing and training points on top of the decision matrix mesh

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)

"""
RBF is the radial basis function. Not sure what that means but it seems to be assuming that those points closest to the spatial mean are most likely to be the point
It seems to work pretty well for the linear data
"""

#%% Exercise 10

# Reset X and y
X, y = np.copy(X_copy), np.copy(y_copy)

X, y = datasets[2] # Choose the linear dataset

# scale the data by the mean and standard deviation, i.e. z = (x - u) / s, 
# where x is the data, u is the mean, and s is the standard deviation
# see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X = StandardScaler().fit_transform(X)

# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

# index for classifier
c = 8  # Choose the RBF SVM classifier

name = names[c] # Use c for the classifier method to allow for modifying of method easily
clf = classifiers[c]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(name)

print(clf)

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance

print(score)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the testing and training points on top of the decision matrix mesh

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)

"""
The Naive Bayes method does alright. It seems to give less weight to points further away so probably is better for datasets that have more overlap
Whatever math happens in the equation seems to have a smaller search distance than some other classifiers
"""

#%% Exercise 11

# Reset X and y
X, y = np.copy(X_copy), np.copy(y_copy)

X, y = datasets[1] # Choose the circular dataset

# scale the data by the mean and standard deviation, i.e. z = (x - u) / s, 
# where x is the data, u is the mean, and s is the standard deviation
# see: https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html
X = StandardScaler().fit_transform(X)

# split the data into training and testing
X_train, X_test, y_train, y_test = \
    train_test_split(X, y, test_size=.4, random_state=42)

# index for classifier
c = 7  # Choose the RBF SVM classifier

name = names[c] # Use c for the classifier method to allow for modifying of method easily
clf = classifiers[c]

clf.fit(X_train, y_train)
score = clf.score(X_test, y_test)

print(name)

print(clf)

# weights = 'uniform' here means we are not weighting by distance
# p=2 is using Euclidean (standard) distance

print(score)

# Plot the decision boundary. For that, we will assign a color to each
# point in the mesh [x_min, x_max]x[y_min, y_max].

if hasattr(clf, "decision_function"):
    Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
else:
    Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]
    
# Put the result into a color plot
Z = Z.reshape(xx.shape)
# note we are using the meshgrid we created before to plot this as a filled contour plot
plt.contourf(xx, yy, Z, cmap=cm, alpha=.8)
plt.colorbar()


# Plot the testing and training points on top of the decision matrix mesh

# Plot the training points
plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
           edgecolors='k')
# Plot the testing points
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
           edgecolors='orange', alpha=1, zorder = 10)

"""
This method does a decent job although it does not seem to be able to definitively predict if any value will be blue where it is more sure about red values farther from the middle
"""

#%% Exercise 11 and beyond. Interpreting the double for loop

# Import many packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

# This goes with the meshgrid setup
h = .02  # step size in the mesh

# Create array with names of different classifiers
names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]

# Creates arrays with different classifier functions
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

# Create a random dataset and add some noise
X, y = make_classification(n_features=2, n_redundant=0, n_informative=2,
                           random_state=1, n_clusters_per_class=1)
rng = np.random.RandomState(2)
X += 2 * rng.uniform(size=X.shape)
linearly_separable = (X, y)

# Create 3 random datasets, one of moon shape, one of circular shape, and one linear
datasets = [make_moons(noise=0.3, random_state=0),
            make_circles(noise=0.2, factor=0.5, random_state=1),
            linearly_separable
            ]

# Define some parameters for figures
figure = plt.figure(figsize=(27, 9), facecolor = 'w')


i = 1 # Counter
# iterate over datasets
"""
The enumerate function helps iterate through a dataset and outputs a count (ds_cnt) and the iterable object (ds)
"""
for ds_cnt, ds in enumerate(datasets):
    # preprocess dataset, split into training and test part
    
    # Define the data
    X, y = ds # Defines a new X and y (not the ones from above) as the dataset X and y columns
    # Standardize the data
    X = StandardScaler().fit_transform(X)
    # Create training and testing data
    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=.4, random_state=42)

    # Create a meshgrid so all plots have same scale
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))

    # just plot the dataset first. I.e. only plot the dataset with no classifiers
    cm = plt.cm.RdBu
    cm_bright = ListedColormap(['#FF0000', '#0000FF'])
    ax = plt.subplot(len(datasets), len(classifiers) + 1, i)
    # Only add title to plot if it is the first dataset (moon dataset)
    if ds_cnt == 0:
        ax.set_title("Input data")
    # Plot the training points
    ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
               edgecolors='k')
    # Plot the testing points
    ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright, alpha=0.6,
               edgecolors='k')
    ax.set_xlim(xx.min(), xx.max())
    ax.set_ylim(yy.min(), yy.max())
    ax.set_xticks(())
    ax.set_yticks(())
    i += 1 # Counter

    # iterate over classifiers
    for name, clf in zip(names, classifiers):
        ax = plt.subplot(len(datasets), len(classifiers) + 1, i) # Create number of subplots with rows = the number of datasets and columns = number of classifiers + 1
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        if hasattr(clf, "decision_function"):
            # Only if the element has the specific attribute decision function
            Z = clf.decision_function(np.c_[xx.ravel(), yy.ravel()])
        else:
            Z = clf.predict_proba(np.c_[xx.ravel(), yy.ravel()])[:, 1]

        # Z is the output predicted values for each classifier using 1 of 2 possible methods

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        ax.contourf(xx, yy, Z, cmap=cm, alpha=.8)

        # Plot the training points
        ax.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=cm_bright,
                   edgecolors='k')
        # Plot the testing points
        ax.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cm_bright,
                   edgecolors='green', alpha=1, zorder = 10)

        ax.set_xlim(xx.min(), xx.max())
        ax.set_ylim(yy.min(), yy.max())
        ax.set_xticks(())
        ax.set_yticks(())
        if ds_cnt == 0:
            # Add a title only if its the first dataset
            ax.set_title(name, fontsize = 14)
        ax.text(xx.max() - .3, yy.min() + .3, ('%.2f' % score).lstrip('0'),
                size=15, horizontalalignment='right')
        i += 1 # Counter

plt.tight_layout()

