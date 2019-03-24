# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:53:11 2019

@author: Miles
"""
#%%
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import pandas as pd

#%%
""" 1) Import as pandas data frame """
#%%
file = "C:\\Users\Miles\Downloads\Must_Sort\movies.xlsx"
movies_sheet1 = pd.read_excel(file, sheet_name=0, index_col=0)
movies_sheet2 = pd.read_excel(file, sheet_name=1, index_col=0)
movies_sheet3 = pd.read_excel(file, sheet_name=2, index_col=0)
movies = pd.concat([movies_sheet1, movies_sheet2, movies_sheet3])
#%%
""" 2) Convert dataframe to xarray dataset """
#%%
moviesdf = xr.Dataset.from_dataframe(movies)
#%%
""" 3) Write dataset as netcdf file """
#%%
moviesdf.to_netcdf("movies.nc")
#%%
""" 4) Load in your .nc file and check it """
#%%
moviesnc = xr.open_dataset('movies.nc')
#%%
""" Assignment for Week 10 Lecture 1 """
#%%
"""
Exercise 1:
    What does it mean that year and duration are negatively correlated? Plot
    those tow variables together and explain.

Answer:
    It means that as the year increases the average duration of movies
    decreases. This is a very low correlation, which is made obvious in the 
    graph. As more movies were made from 1960 onwards the majority of them
    were not longer than 130 miutes.
"""
#%% Plot budgets vs. Gross Earnings
plt.scatter(movies['Year'], movies['Duration'])
plt.xlabel('Year')
plt.ylabel('Duration')
#%%
"""
Exercise 2:
    Which pairs of variables have the highest correlation? What might explain
    this?

Answer:
    'Facebook Likes by Lead Actor' and 'Facebook Likes for the Whole Cast.'
    This would indicate that the majority of the sway on 'Facebook like for
    the Whole Cast' are being strongly influenced by the popularity of the 
    lead actor in the film, as a popular actor would likely have more Facebook
    likes than an actor with lower popularity.
"""
#%%
"""
Exercise 3:
    Do the tutorial.
    
Answer:
    Within my 'Notes' for Week 10.
"""
#%%
"""
Exercise 4:
    Following the methods in the tutorial above, do the regression with just
    the rooms and not the other variable. How does this compare with the scipy
    result?
Answer:
    Clearly room numbers are not the only determinor of house prices.
"""
#%% Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from scipy import stats
import statsmodels.api as sm
#%% Loads the housing data from the scikit-learn library
from sklearn.datasets import load_boston
boston_dataset = load_boston()
#%% MEDV is the feature variable we are interested in
boston_dataset.DESCR
#%% Put the target (housing value ==MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
#%% Creates multiple variables and we provide names for those variables, which saves them
slope, intercept, r_value, p_value, std_err = stats.linregress(df["RM"], target["MEDV"])
#%%
X = df["RM"] # what we think the cost depends on
y = target["MEDV"]
#%% Note: the target goes first
model = sm.OLS(y, X).fit()
#%% this fits y = aX with no constant (0 intercept)
model.summary()
#%% make the predictions by the model
predictions = model.predict(X)
#%% model with NOT a great fit
plt.plot(X, predictions, 'k--')
plt.scatter(df["RM"], target["MEDV"])
plt.xlabel('Number of Rooms')
plt.ylabel('House Value ($1,000s)')
#%%
"""
Geopanda Bullshit
"""
#%%
import geopandas as gpd
#%%
fp = "C:\\Users\Miles\Downloads\EEZ_land_union_v2_201410\EEZ_land_v2_201410.shp"
#%%
data = gpd.read_file(fp)
#%%
data.plot()
#%%
data.crs
