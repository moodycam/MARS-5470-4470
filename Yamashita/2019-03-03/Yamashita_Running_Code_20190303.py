# -*- coding: utf-8 -*-
"""
Created on Sat Mar  2 14:59:55 2019

@author: tomyamashita
"""

#%% 2/25/2019 Opening exercise

import numpy as np
import matplotlib.pyplot as plt
import xarray as xr

#%% Warm up exercise 1

lat = np.arange(38, 43, 1)
lon = np.arange(-108, -103, 1)

lon_mesh, lat_mesh = np.meshgrid(lon,lat)

H = -100*(lat_mesh -40)**2 - 400*(lon_mesh + 106)**2 + 8000

p = plt.pcolormesh(lon,lat,H, cmap = "terrain")
plt.colorbar(p)
plt.title("Colorado Rockies Heights", fontsize = 16)
plt.contour(lon, lat, H, cmap = "Greys")

#%% Using contour

plt.contour(lon, lat, H)
plt.contourf(lon, lat, H)  # This is a filled version of contour lines

#%% Messing around with number of values

lat2 = np.arange(38, 43, .01)
lon2 = np.arange(-108, -103, .01)

lon_mesh2, lat_mesh2 = np.meshgrid(lon2,lat2)

H2 = -100*(lat_mesh2 -40)**2 - 400*(lon_mesh2 + 106)**2 + 8000

p2 = plt.pcolormesh(lon2,lat2,H2, cmap = "terrain")
plt.colorbar(p2)
plt.title("Colorado Rockies Heights", fontsize = 16)
plt.contour(lon2,lat2,H2, cmap = "Greys", levels = 16) #250 foot contour lines


#%% Going over meshgrids again

"""
Mesh grids fill values across an axis in a mulit-dimensional array based on a 1D array
"""

#%% File input/output lecture

# Loading text files
population = np.loadtxt("populations.txt")
population

# Saving text files
# np.savetxt("test.txt", population)

# NetCDF files
Net_data = xr.open_dataset("CESM.003.SST.1980.nc")
Net_data

# Can extract variables from a netcdf file
lat_test = np.array(Net_data.lat)
lon = np.array(Net_data.lon)

# There are different packages for working with different types of data 
# Before xarray, there was netCDF4
import netCDF4 as nc

data2 = nc.Dataset("CESM.003.SST.1980.nc", 'r') # the 'r' means to just read the file
data2
data2.variables
data2.dimensions
sst2 = data2.variables['SST'][:]

#%% Spreadsheets

import pandas as pd

movies = pd.read_excel("movies.xls")
help(movies)

# Pandas imports data as data frames

# Can change this from a pandas data fram to an xarray dataset
movies2 = xr.Dataset.from_dataframe(movies)

#%% Exercises E2

population = np.loadtxt("populations.txt")
np.savetxt("test.txt", population)

"""
No they look very different
The original text file was a single line with a lot of data across
The values are the same
The new file is set up like a table and doesn't use scientific notation
Also, its values go out to a lot of points
Also, the table headers are missing in the new file
"""

#%% Exercise 3

np.savetxt("test_header.txt", population, fmt = '%.0f', header = "Population data for animals. Columns are Year, Hares, Lynx, Carrots")
# To change the format of the numbers use the fmt object:
    # e indicates scientific notation, f is no scientific notation
    # % = value, .0 = no decimal places, f = no scientific notation

#%% Exercise 4

import xarray as xr
data = xr.open_dataset("CESM.003.SST.1980.nc")
SST = np.array(data.SST)

import netCDF4 as nc
data2 = nc.Dataset("CESM.003.SST.1980.nc", 'r')
SST2 = data2.variables['SST'][:]

"""
SST2 is a masked array
SST2 outputs the SST data as a masked array where all nan values are masked as false
The values that are numbers are kept, while nans are excluded
"""

#%% Exercise 5

SST2 # From netCDF4
SST # From xarray
SST_mask = SST < 100000 # Create mask of data you want to keep
# Use logical_not to keep data you want to keep
SST3 = np.ma.masked_array(SST, np.logical_not(SST_mask))

SST4 = np.ma.masked_invalid(SST)

"""This doesn't work
SST5_mask = SST == np.nan
SST5 = np.ma.masked_where(SST == np.nan, SST)
"""

#%% Exercise 6

import pandas as pd

excel_file = 'movies.xls'
# This will read the first sheet of an excel file. Can specify different sheets
movies = pd.read_excel(excel_file)

movies.head()

# Index_col specifies which column is the index column
# sheet_name specifies which sheet to use
movies_sheet1 = pd.read_excel(excel_file, sheet_name = 0, index_col = 0)
movies_sheet1.head()
movies_sheet2 = pd.read_excel(excel_file, sheet_name = 1, index_col = 0)
movies_sheet2.head()
movies_sheet3 = pd.read_excel(excel_file, sheet_name = 2, index_col = 0)
movies_sheet3.head()

# concat can be used to combine dataframes into a single data frame
movies = pd.concat([movies_sheet1, movies_sheet2, movies_sheet3])
movies.shape

#%% Using excelfile to read multiple sheets

# Excel file is an easier way to import excel data that has a lot of sheets where you need the data from all of them


xlsx = pd.ExcelFile(excel_file)
movies_sheets =[]
for sheet in xlsx.sheet_names:
    movies_sheets.append(xlsx.parse(sheet))
movies = pd.concat(movies_sheets)
movies.shape
movies.tail # this can view the bottom rows of the sheets

#%% Sorting data

sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending = False)
sorted_by_gross["Gross Earnings"].head

#%% Plotting data

import matplotlib.pyplot as plt

sorted_by_gross['Gross Earnings'].head(10).plot(kind = "barh")
plt.show()

movies['IMDB Score'].plot(kind = "hist")
plt.show

#%% Statistical info from data

movies.describe()

# can read basic statistical information from the data
movies['Gross Earnings'].mean()

"""
Since this dataset isn't available on the site...
movies_skip_rows = pd.read_excel("movies-no-header-skip-rows.xls", header = None, skiprows = 4)
movies_skip_rows.head(5)
movies_skip_rows.columns = ['Title', 'Year', 'Genres', 'Language', 'Country', 'Content Rating',
       'Duration', 'Aspect Ratio', 'Budget', 'Gross Earnings', 'Director',
       'Actor 1', 'Actor 2', 'Actor 3', 'Facebook Likes - Director',
       'Facebook Likes - Actor 1', 'Facebook Likes - Actor 2',
       'Facebook Likes - Actor 3', 'Facebook Likes - cast Total',
       'Facebook likes - Movie', 'Facenumber in posters', 'User Votes',
       'Reviews by Users', 'Reviews by Crtiics', 'IMDB Score']
movies_skip_rows.head()
"""

# Read the first 6 columns of the dataset
movies_subset_columns = pd.read_excel(excel_file, usecols = 6)
movies_subset_columns.head()

#%% Applying formulas to columns

# Create new column that contains data compiled from basic arithmetic on other columns
movies["Net Earnings"] = movies["Gross Earnings"] - movies["Budget"]
sorted_movies = movies[["Net Earnings"]].sort_values(['Net Earnings'], ascending = [False])
sorted_movies.head(10)['Net Earnings'].plot.barh()
plt.show()

#%% Pivot Table in Pandas

# Identify an index and a column to summarize by
movies_subset = movies[['Year', 'Gross Earnings']]
movies_subset.head()

# Create pivot table of the data
# Need to specify an index but when only 2 columns, don't need to specify summary value
# Default pivot method is SUM
# This seems to excluding NaNs when creating pivot table while same code in tutorial includes NaNs
earnings_by_year = movies_subset.pivot_table(index = ['Year'])
earnings_by_year.head(25)
earnings_by_year.plot()
plt.show()

movies_subset = movies[['Country', 'Language', 'Gross Earnings']]
movies_subset.head()

earnings_by_co_lang = movies_subset.pivot_table(index = ['Country', 'Language'])
earnings_by_co_lang.head()
earnings_by_co_lang.head(20).plot(kind = 'bar', figsize = (12,4))
plt.show()

#%% Exporting to excel

movies.to_excel('output.xlsx')

"""
In this version of movies that I am currently working in, the index for movies is the title
Anyway, can exclude the index when it is just an OID number using the following syntax: 
movies.to_excel('output.xlsx', index=False)
"""

# Can customize the excel output using a writer
writer = pd.ExcelWriter('output2.xlsx', engine = 'xlsxwriter')
movies.to_excel(writer, index=True, sheet_name = 'report')
workbook = writer.book
worksheet = writer.sheets['report']

# Setting the header format to bold
header_fmt = workbook.add_format({'bold': True})
worksheet.set_row(0, None, header_fmt)
writer.save()

# Can set various other formatting to the excel file using this function

#%% 2/27/2019

"""
Tutorial is finished above so exercises...

Exercise 1
From the movies spreadsheet in the above tutorial, make a histogram of 
all the years movies were made, combining the data in three sheets. 
Make two plots, one with bins (bars) of 10 years width, and one with bins of 20 years width. 
Label your axes and change the color of the bars. 
Interpret this data, i.e. describe the change in the number of movies made over time.

Exercise 2
Make a scatterplot of budget vs. year (x-axis), again combining data from the 
three sheets. Label your axes. What kind of trend is there in this data, 
how do budgets change over time? Is the relationship linear?

Exercise 3
Make a scatterplot of budget vs. gross earnings for all of the data. 
Describe the relationship between these variables, are they correlated?

Exercise 4
What is the lowest IMDB score? Find this using the programming techniques 
we learned in class

Exercise 5
Find the names of all the movies with an IMDB score below 5 and put them in an 
array or list, with an associated array of their IMDB scores. 
Write these titles and scores to a text file with the appropriate header information
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

#%% Exercise 1

excel_file = pd.ExcelFile("movies.xls")
movies_sheets = []
for sheet in excel_file.sheet_names:
    movies_sheets.append(excel_file.parse(sheet))
movies = pd.concat(movies_sheets)

movies['Year'].head()

fig, ax = plt.subplots(figsize = (12,4), ncols = 2, sharey = True)
ax[0].hist(movies['Year'], bins = 20, color = "red")
ax[1].hist(movies['Year'], bins = 10, color = "violet")
ax[0].set_title("10 year bins")
ax[1].set_title("20 year bins")
ax[0].set_ylabel("Number of movies", fontsize = 14)
fig.text(0.50, 0.00, "Year", fontsize = 14)
fig.suptitle("Histogram of years of movies", fontsize = 16)
print("The number of movies has increased dramatically over time especially since around 1990")

#%% Exercise 2

excel_file = pd.ExcelFile("movies.xls")
movies_sheets = []
for sheet in excel_file.sheet_names:
    movies_sheets.append(excel_file.parse(sheet))
movies = pd.concat(movies_sheets)

year = movies['Year']
budget = movies['Budget']

budget.max

fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(year, budget)
ax.set_ylim(0,500000000)
ax.set_xlabel("Year")
ax.set_ylabel("Budget in $100,000,000")
print("Movie budgets seem to exponentially increase over time")

# No pearson correlation coefficient can be obtained from this data
from scipy.stats.stats import pearsonr
print(pearsonr(year, budget))

#%% Exercise 3

excel_file = pd.ExcelFile("movies.xls")
movies_sheets = []
for sheet in excel_file.sheet_names:
    movies_sheets.append(excel_file.parse(sheet))
movies = pd.concat(movies_sheets)

budget = movies['Budget']
gross = movies['Gross Earnings']

fig, ax = plt.subplots(figsize = (6,4))
ax.scatter(budget, gross)
ax.set_xlabel("Budget")
ax.set_ylabel("Gross Earnings")
ax.set_xlim(0,500000000)

# Correlation can't be calculated because there are many many NaNs
print(np.corrcoef(budget, gross))

print("Budget and gross earnings do not seem to show much correlation between them. Maybe a slight positive correlation but not much")

#%%Exercise 4

excel_file = pd.ExcelFile("movies.xls")
movies_sheets = []
for sheet in excel_file.sheet_names:
    movies_sheets.append(excel_file.parse(sheet))
movies = pd.concat(movies_sheets)

IMDB = movies['IMDB Score']
min(IMDB)

#%% Exercise 5

excel_file = pd.ExcelFile("movies.xls")
movies_sheets = []
for sheet in excel_file.sheet_names:
    movies_sheets.append(excel_file.parse(sheet))
movies = pd.concat(movies_sheets)

IMDB = movies['IMDB Score']
names = movies['Title']

movies_sub = movies[IMDB < 5]

bad_movies = np.array([movies_sub['Title'], movies_sub['IMDB Score']])
bad_movies.T

bad_movies2 = pd.DataFrame(bad_movies.T, columns = ("Title", "IMDB Score"))

# Text files cannot write strings
# np.savetxt("bad_movies.txt", bad_movies.T, fmt = '%.0s')

bad_movies2.to_csv("Bad_movies.csv", index = False)


#%% Project related things

"""
Recreating emerging hot spot analysis in python

Look up source code for arcgis version of the analysis then recreate in an open source environment

Python packages that will be required = 
pysal (Spatial analysis package)
scipy.stats (contains Mann-Kendall Time Series function)

"""
