# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:37:27 2019

@author: Miles
"""

#%% Classic packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
#%%
"""
Warm Up Exercise 1
""""
#%% Step 1
lat = np.linspace(38,42,100)
lon = np.linspace(-108, -104, 100)
#%% Step 2
Lon, Lat = np.meshgrid(lon, lat)
#%% Step 3
plt.pcolormesh(Lat)
plt.colorbar()
#%% Step 4
plt.pcolormesh(Lon)
plt.colorbar()
#%% Step 5
H = -100*((Lat-40)**2) - 400*((Lon+106)**2) + 8000
#%% Step 6
np.shape(H)
#%% Step 7
plt.pcolormesh(H)
plt.colorbar()
#%% Step 8
plt.contour(H)
plt.colorbar()
#%% Step 9
""" Location is Jacobs Ladder, Colorado """
#%%
"""
Lecture 7.2 Exercises
"""
#%% Read Data from the Excel file
excel_file = "C:\\Users\Miles\Downloads\movies.xls"
movies = pd.read_excel(excel_file)
#%%
movies.head()
#%%
movies_sheet1 = pd.read_excel(excel_file, sheetname=0, index_col=0)
movies_sheet1.head()
#%%
movies_sheet2 = pd.read_excel(excel_file, sheetname=1, index_col=0)
movies_sheet2.head()
#%%
movies_sheet3 = pd.read_excel(excel_file, sheetname=2, index_col=0)
movies_sheet3.head()
#%%
movies = pd.concat([movies_sheet1, movies_sheet2, movies_sheet3])
movies.head()
#%%
movies.shape
#%%
""" Ex. 1 """
# From the movies spreadsheet in the above tutorial, make a histogram of all the
# years movies were made, combining the data in the three spreadsheets. Make
# two plots, one with bins of 10 year width and one with bins of 20 year
# width. Label your axes and change the color of the bars. Interpret the data.
#%%
plt.hist(movies['Year'], bins=[1910, 1920, 1930, 1940, 1950, 1960, 1970, 1980, 1990, 2000, 2010, 2020], color='grey')
plt.suptitle('Movies by Year (10 Year Bins)', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
#%%
plt.hist(movies['Year'], bins=[1910, 1920, 1940, 1960, 1980, 2000, 2020], color='grey')
plt.suptitle('Movies by Year (20 Year Bins)', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Frequency', fontsize=16)
#%%
"""
The number of movies made over time is clearly increasing at an exponential
rate. More than likely this is due to advances in technology and greater
appretiation for film making as an artistic medium. It could likely be argued
to be the most influential artistic medium of the 2000s.
"""
#%%
""" Ex. 2 """
# Make a scatterplot of budget vs. year (x-axis), again combining data from the
# three sheets. Label your axes. What king of trend is there in this data,
# how do budgets change over time? Is the relationship linear?
#%%
plt.scatter(movies['Year'], movies['Budget'], color='black')
plt.ylim(0, 450000000)
plt.xlim(1910, 2017)
plt.suptitle('Movie Budgets by Year', fontsize=20)
plt.xlabel('Year', fontsize=16)
plt.ylabel('Budget (Hundred Millions)', fontsize=16)
#%%
"""
The relationship is NOT linear, it is exponential.
"""
#%%
""" Ex. 3 """
# Make a scatterplot of budget vs. gross earning for all of the data. Describe
# the relationship, are the variables correlated?
#%%
plt.scatter(movies['Budget'], movies['Gross Earnings'], color='black')
# plt.ylim(0, 450000000)
plt.xlim(0, 450000000)
plt.ylim(0, 800000000)
plt.suptitle('Movie Earnings by Budget', fontsize=20)
plt.xlabel('Budget (Hundred Millions)', fontsize=16)
plt.ylabel('Earnings (Hundred Millions)', fontsize=16)
#%%
"""
These two variables do appear to be slightly correlated. There is some noise
at many of the lower budgets, with some films doing extremely well on low
budgets while others do quite poor at high budgets, but overall they appear
to have a slight linear correlation. If there is a correlation it is very
low.
"""
#%%
""" Ex. 4 """
# What is the lowest IMDB score? Find this using the programming techniques
# we learned in class.
#%%
movies['IMDB Score'].min()
#%%
"""
The lowest scoring movie on IMDB was "Justin Bieber: Never Say Never", which
got a 1.6.
"""
#%%
""" Ex. 5 """
# Find the names of all the moves with an IMDB score below 5 and put them in
# an array or list, with an associated array of their IMDB scores. Write
# these title and scores to a text file with the appropriate header info.
#%%
shitty_movies = movies[movies['IMDB Score'] <= 5]
#%%
shitty_movies.head
#%%
shitty_movies.to_excel('shitty_movies.xlsx')