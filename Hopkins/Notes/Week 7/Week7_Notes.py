# -*- coding: utf-8 -*-
"""
Created on Mon Feb 25 13:37:18 2019

@author: Miles
"""
#%%
"""
File Input/Output (I/O)
"""
#%% Classic packages
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
import matplotlib as mpl
import netCDF4 as nc # For netCDF files
import pandas as pd # For spreadsheets
#%%
# import OpenPyXL # Read/Write Excel 2010 xlsx/xlsm files
import xlrd # Read Excel data
import xlwt # Write to Excel
# import XlsxWriter # Write to Excel (xlsx) files
#%%
tdata = np.loadtxt('C:\\Users\Miles\Downloads\populations.txt')
#%%
tdata
#%%
""" Exercises """
#%% E2
"""
E2: Go look at the file that was just created. Is it the same as
"populations.txt"? If not, how is it different?
Answer: It is not the same. The original file had a header, while the new file
we created lost its header.
"""
#%% E3
"""
E3: Add a header to this file (see the function documentation) with the names
of the variables.
Answer: See code below
""""
np.savetxt('test.txt', tdata, fmt = "%.0f", header = "Year Hare Lynx Carrot")
#%%
""" Netcdf Files """
#%%
file = 'C:\\Users\Miles\Downloads\CESM.003.SST.1980.nc'
#%%
data = xr.open_dataset(file)
#%%
""" Note: data is an xarray dataset, which gives you a bunch of "metadata"
about what is in the netcdf file """
data
#%%
lat = np.array(data.lat) # Extracts files from the above netcdf file
lon = np.array(data.lon) # Does lon
sst = np.array(data.SST) # Grabs monthly SST
"""
xarray is built on top of the nedCDF4 package
"""
#%%
data2 = nc.Dataset(file, 'r') # 'r' is for read
data2
#%%
data2.variables
#%%
data2.dimensions
#%%
sst2 = data2.variables['SST'][:]
#%%%
type(sst2)
#%%
sst2
#%%
type(sst)
#%%
sst
#%% E4
"""
E4: What type of object is "sst2"? Is it different from "sst" we got with
xarray and numpy? How?
Answer: Yes. "sst2" is a 'numpy.ma.core.MaskedArray' while "sst" is an
'numpy.ndarray'. To make sst like sst2 we would have to mask it.
"""
#%% E5
"""
E5: Make sst like sst2. Hint, this involves masking
Answer: See code below
"""
sst1 = np.ma.masked_invalid(sst) # Masks everything except
    ## the bits you want to keep
#%%
sst1
#%%
""" Spreadsheets """
#%% For Excel Files!!!
file = "C:\\Users\Miles\Downloads\movies.xls"
movies = pd.read_excel(file) # This is a pandas dataframe
#%%
help(movies)
#%% Can change the panda file to an xarray dataset if you like
# http://xarray.pydata.org/en/stable/pandas.html
movies2 = xr.Dataset.from_dataframe(movies)
movies2
#%%
movies.Year
#%% E6
"""
E6 (rest of lab): Go through this tutorial on excel and pandas
"""
#%%
# https://www.dataquest.io/blog/excel-and-pandas/
#%%
""" Read data from the Excel file """
excel_file = "C:\\Users\Miles\Downloads\movies.xls"
movies = pd.read_excel(excel_file)
#%%
movies.head()
#%% Inputs Sheet 1
movies_sheet1 = pd.read_excel(excel_file, sheetname=0, index_col=0)
movies_sheet1.head()
#%% Inputs Sheet 2
movies_sheet2 = pd.read_excel(excel_file, sheetname=1, index_col=0)
movies_sheet2.head()
#%% Inputs Sheet 3
movies_sheet3 = pd.read_excel(excel_file, sheetname=2, index_col=0)
movies_sheet3.head()
#%% Combines all Sheets together
movies = pd.concat([movies_sheet1, movies_sheet2, movies_sheet3])
#%%
movies.shape
#%%
""" Using the ExcelFile class to read multiple sheets """
#%%
xlsx = pd.ExcelFile(excel_file)
movies_sheets = []
for sheet in xlsx.sheet_names:
    movies_sheets.append(xlsx.parse(sheet))
movies = pd.concat(movies_sheets)
#%%
movies.head # Shows the first few rows
#%%
""" Exploring the Data """
#%%
movies.shape # Gives the shape
#%%
movies.tail()
#%%
sorted_by_gross = movies.sort_values(['Gross Earnings'], ascending = False)
sorted_by_gross["Gross Earnings"].head(10) # Top 10 gross earnings
#%%
sorted_by_gross['Gross Earnings'].head(10).plot(kind="barh") # Look at all those periods!
plt.show() # Shows a great bar graph
#%%
movies['IMDB Score'].plot(kind="hist")
plt.show()
#%%
""" Getting statistical information about the data """
#%%
movies.describe()
"""
Displays below information:
    the count or number of values
    mean
    standard deviation
    min and max
    25%, 50%, and 75% quantile
"""
#%%
movies["Gross Earnings"].mean()
#%%
""" Reading files with no header and skipping records """
#%%
movies_skip_rows = pd.read_excel("movies-no-header-skip-rows.xlsx", header=None, skiprows=4)
movies_skip_rows.head(5)
#%%
movies_skip_rows.columns = ['Title', 'Year', 'Genres', 'Language', 'Country', 'Content Rating',
       'Duration', 'Aspect Ratio', 'Budget', 'Gross Earnings', 'Director',
       'Actor 1', 'Actor 2', 'Actor 3', 'Facebook Likes - Director',
       'Facebook Likes - Actor 1', 'Facebook Likes - Actor 2',
       'Facebook Likes - Actor 3', 'Facebook Likes - cast Total',
       'Facebook likes - Movie', 'Facenumber in posters', 'User Votes',
       'Reviews by Users', 'Reviews by Crtiics', 'IMDB Score']
movies_skip_rows.head()
#%%
""" Reading a Subset of Columns """
#%%
movies_subset_columns = pd.read_excel(excel_file, parse_cols=6)
movies_subset_columns.head()
#%%
movies["Net Earnings"] = movies["Gross Earnings"] - movies["Budget"]
#%%
sorted_movies = movies[['Net Earnings']].sort_values(['Net Earnings'], ascending=[False])
sorted_movies.head(10)['Net Earnings'].plot.barh()
plt.show()
#%%
""" Pivot Table in Pandas """
#%%
movies_subset = movies[['Year', 'Gross Earnings']]
movies_subset.head()
#%%
earnings_by_year = movies_subset.pivot_table(index=['Year'])
earnings_by_year.head()
#%%
earnings_by_year.plot()
plt.show()
#%%
movies_subset = movies[['Country', 'Language', 'Gross Earnings']]
movies_subset.head()
#%%
earnings_by_co_lang = movies_subset.pivot_table(index=['Country', 'Language'])
earnings_by_co_lang.head()
#%%
earnings_by_co_lang.head(20).plot(kind='bar', figsize=(20,8))
plt.show()
#%%
""" Exploring the Results to Excel """
#%%
movies.to_excel('output.xlsx')
movies.head()
#%%
movies.to_excel('output.xlsx', index=False)
#%%
writer = pd.ExcelWriter('output.xlsx', engine='xlsxwriter')
movies.to_excel(writer, index=False, sheet_name='report')
workbook = writer.book
worksheet= writer.sheets['report']
#%%
header_fmt = workbook.add_format({'bold': True})
worksheet.set_row(0, None, header_fmt)
#%%
writer.save()
