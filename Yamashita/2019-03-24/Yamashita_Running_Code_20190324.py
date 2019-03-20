# -*- coding: utf-8 -*-
"""
Created on Wed Mar 20 17:18:33 2019

@author: tjyamashta_dell
"""

#%% 3/18/2019 Correlation and Linear Regression

# Warm up exercise

import numpy as np
import xarray as xr
import pandas as pd
import netcdf4 as nc4

excel_file = pd.ExcelFile("movies.xls")
movies_sheets = []
for sheet in excel_file.sheet_names:
    movies_sheets.append(excel_file.parse(sheet))
movies = pd.concat(movies_sheets)

# Write pandas dataframe as an xarray dataset
movies_nc = xr.Dataset.from_dataframe(movies)

# Need to define dimensions
# Only 1 dimension in this dataset = Year
# Everything else is dimensionless?

# Write xarray dataset as a netcdf file?
# Does something weird and doesn't create a netcdf file with dimensions

# This is the same as the un-text version...
# movies_nc2 = xr.Dataset.to_netcdf(movies_nc, "movies.nc")

movies_nc.to_netcdf("movies2.nc")

movies_test = xr.open_dataset("movies.nc")
movies_test

movies_test3 = xr.open_dataset("movies2.nc")
movies_test3


#%% Lecture

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

#%% Correlation

excel_file = pd.ExcelFile("movies.xls")
movies_sheets = []
for sheet in excel_file.sheet_names:
    movies_sheets.append(excel_file.parse(sheet))
movies = pd.concat(movies_sheets)

# Plot 1. Scatterplot between Budget and Gross Earnings
plt.scatter(movies['Budget'], movies['Gross Earnings'])
plt.plot([0,1E10], [0, 1E10], 'k--')  # Creates black dashed line plot showing perfect positive correlation
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')
plt.xlim([0, 1E9])
plt.ylim([0, 1E9])

movies_corr = movies.corr()
movies_corr

#%% Linear regression

from sklearn import datasets
data = datasets.load_boston()
#data?

print(data.DESCR)

df = pd.DataFrame(data.data, columns=data.feature_names)
target = pd.DataFrame(data.target, columns=["MEDV"])

print(np.corrcoef(df["RM"], target["MEDV"]))

plt.scatter(df["RM"], target["MEDV"])
plt.xlabel('Number of Rooms')
plt.ylabel("House Value ($1000s)")

#%% Linear regression with scipy

from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(df["RM"], target["MEDV"])

print(slope)
print(intercept)
print(r_value)
print(p_value)
print(std_err)

plt.plot(df["RM"], slope*df['RM']+intercept, 'k--')
plt.scatter(df["RM"], target["MEDV"])
plt.xlabel('Number of Rooms')
plt.ylabel("House Value ($1000s)")

#%% Linear regression with statsmodels

import statsmodels.api as sm

X = df["RM"]
y = target['MEDV']

model = sm.OLS(y, X).fit()

print(model.summary())

predictions = model.predict(X)

plt.plot(X, predictions, 'k--')
plt.scatter(X, y)
plt.xlabel("Number of Rooms")
plt.ylabel("House Value ($1000s)")

#%% Exercise 1

"""
Year and Duration are negatively correlated (-0.13) indicating that movies have gotten shorter over time
There is very little correlation between year and duration so this relationship is probably not significant
The scatterplot indicates that there is very little relationship with newer movies both being a lot longer and a lot shorter than older movies
This variation is probably the cause of the correlation seen
"""

plt.scatter(movies['Year'], movies['Duration'])
plt.xlabel('Year')
plt.ylabel('Duration')

#%% Exercise 2

"""
The maximum correlation between 2 different variables is between Facebook Likes (actor 1) and Facebook Likes (cast total)
Correlation = 0.95
This is probably because total likes for the cast is heavily driven by likes for the first actor
indicating that these 2 data are not independent and one should get thrown out. 
Also this is a pretty useless correlation
"""

#%% Exercise 3

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from sklearn.datasets import load_boston
boston_dataset = load_boston()

print(boston_dataset.keys())
boston_dataset.DESCR

boston = pd.DataFrame(boston_dataset.data, columns = boston_dataset.feature_names)
boston.head()

boston['MEDV'] = boston_dataset.target

boston.isnull().sum()

#%% Exploratory data analysis

sns.set(rc={'figure.figsize':(11.7, 8.27)})
sns.distplot(boston['MEDV'], bins = 30)
plt.show()

correlation_matrix = boston.corr().round(2)

sns.heatmap(data=correlation_matrix, annot=True, cmap = "seismic")

plt.figure(figsize=(20,5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = boston[col]
    y = target
    plt.scatter(x, y, marker = "o")
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
    
#%% Preparing data for training the model

X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns = ['LSTAT', 'RM'])
Y = boston['MEDV']

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)

#%% Training and testing the model

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)

#%% Model evaluation

# model evaluation for the training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")

# model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for testing set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))

#%% Exercise 4

# Scipy Results. This might not work if everything is run in order because something in the tutorial changes the target["MEDV"]
from scipy import stats
slope, intercept, r_value, p_value, std_err = stats.linregress(df["RM"], target["MEDV"])

print('Scipy Slope is {}'.format(slope))
print('Scipy Intercept is {}'.format(intercept))
print('Scipy R^2 value is {}'.format(r_value))
print('Scipy p value is {}'.format(p_value))
print('Scipy Standard error is {}'.format(std_err))

# sklearn results
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score


X2 = pd.DataFrame(boston['RM'], columns = ['RM'])
Y2 = boston['MEDV']

X2_train, X2_test, Y2_train, Y2_test = train_test_split(X2, Y2, test_size = 0.2, random_state=5)
lin_model2 = LinearRegression()
lin_model2.fit(X2_train, Y2_train)

y_train_predict2 = lin_model2.predict(X2_train)
rmse_train = (np.sqrt(mean_squared_error(Y2_train, y_train_predict2)))
r2_train = r2_score(Y2_train, y_train_predict2)

y_test_predict2 = lin_model2.predict(X2_test)
rmse_test = (np.sqrt(mean_squared_error(Y2_test, y_test_predict2)))
r2_test = r2_score(Y2_test, y_test_predict2)

print('Training RMSE is {}'.format(rmse_train))
print('Training R2 score is {}'.format(r2_train))

print('Test RMSE is {}'.format(rmse_test))
print('Test R2 score is {}'.format(r2_test))

"""
The Scipy results and the sklearn test results were very similar. 
Only different beyond the hundredths place
"""

#%% 3/20/2019

"""
Lecture from Cameron on Scrum
Project Management
"""

"""
Overleaf and LaTeX and Zotero
"""

"""
Creating a new Python2 Environment
"""

#%% Geopandas

import geopandas as gp
import matplotlib.pyplot as plt

roads = gp.read_file('txdot-2015-roadways_tx.shp')

roads.head()

#roads.plot(column = 'RTE_CLASS')

#%% Highways only map

highways = roads[roads.RTE_CLASS == 'On System Highways'] 

highways.head()
highways.plot()

#%% Write to shapefile

gp.GeoDataFrame.to_file(highways, "highways_tx.shp")
