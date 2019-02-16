# -*- coding: utf-8 -*-
"""
Created on Sat Feb 16 10:33:09 2019

@author: tomyamashita
"""

"""
Lecture related things for the week of 2/11/2019
2/13/19 notes begins approximately on line: 307
"""

#%% 2/11/2019 Riley's Lecture again

# Installing xarray and cartopy
# Can do from either the anaconda navigator or from anaconda prompt
# From prompt: 
    # conda install (package name)

#%% Now Riley's Lecture

# Need to import packages 
import numpy as np
import matplotlib.pyplot as plt

# View plots within console line
%matplotlib inline

#%% Setting up plots

# figsize can be used to change the size of the plot
f = plt.figure(figsize=(8,6))
plt.subplot
# fig controls figure properties
# ax controls axis properties
fig, ax = plt.subplots()
fig, ax = plt.subplots(figsize=(12,8), nrows=2, ncols=2)

#%% Line plots

x = np.linspace(-np.pi, np.pi, 64, endpoint=True)
y = np.cos(x)
plt.plot(y)
plt.plot(x,y)

# Can also use the scatter command which creates scatter plots

#%% Default line plots

x = np.linspace(-np.pi, np.pi, 64, endpoint=True)
C, S = np.cos(X), np.sin(X)

fig, ax = plt.subplots()
ax.plot(X, C)
ax.plot(X, S)

#%% Multiplots

fig, ax = plt.subplots(figsize=(8,3), ncols=2, sharex=True, sharey=True)
# Sharex and sharey allow multiplots to share axes when they are the same
ax[0].plot(X,C, label = 'cosine')
ax[0].plot(X,S, label = 'sine')
ax[1].plot(X,S, label = 'sine')
# Can add a legend to plot when there are multiple data points in the same graph
ax[0].legend()

#%% Modifying plot aesthetics

"""
Can use different keywords to modify properties of the plot
color = modify color of the line. Defaults to blue
linewidth = change width of the line. Defaults to 1
linestyle = style of the line being plotted. Defaults to '-'
marker = type fo marker to plot over the line
"""

fig, ax = plt.subplots()
ax.plot(X, C, label = 'cosine', linewidth=3, color="green")
ax.plot(X, S, label = 'sine', linewidth=1.5, linestyle='-.', color = "purple")

"""
Can also set axes limits, titles, and labels
"""

"""
Can use style sheets as well to set the aesthetics of graphs
These pull from commonly seen graph styles and uses those settings for graphs
"""

#%% Meshes (Finally onto stuff I didn't do last time)

# To import netcdf files make sure you have the netcdf4 package installed
import xarray as xr
# Can use open_dataset() to open netcdf files
data = xr.open_dataset('CESM.003.SST.1980.nc')
# Create variables from different columns of data
lat = np.array(data.lat)
long = np.array(data.lon)
data = np.array(data.SST)
print(data.shape)
meandata = np.mean(data, axis=0)
anom = data - meandata

#%% Basic Pcolor map
fig, ax = plt.subplots()
p = ax.pcolormesh(long, lat, meandata, cmap = "bwr")
cb = plt.colorbar(p)

# Can use cmap in the pcolormesh function to change the color map

# Aesthetics
# The piece in [] creates special characters such as superscript and subscript
cb.set_label('Sea Surface Tempearture [$^{o}$C]')
ax.set_title('CESM Sea Surface Temperature (1970-1980)')
ax.set_ylabel('Latitude')
ax.set_xlabel('Longidtude')

#%% Geography

# Import things from cartopy 
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.util import add_cyclic_point

#%% 

fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Robinson()))
ax.add_feature(cfeature.LAND, color='k')
ax.pcolormesh(long, lat, meandata, transform=ccrs.PlateCarree())

#%% Seam issue

# Apparently There is a seam issue when data wraps around the world
cyclic_data, cyclic_lons = add_cyclic_point(meandata, coord=long)
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Robinson()))
ax.add_feature(cfeature.LAND, color='k')
p= ax.pcolormesh(cyclic_lons, lat, cyclic_data, transform=ccrs.PlateCarree(), cmap="plasma")
plt.colorbar(p, orientation='horizontal', pad=0.05, fraction=0.05, extend='both')
# pad is the distance between the colorbar and the map
# fraction is the size of the colorbar in relation to the map

#%% Different projections
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.Geostationary()))
# Can use the zorder function to force features to be on top
ax.add_feature(cfeature.LAND, color='k', zorder=4)
p= ax.pcolormesh(cyclic_lons, lat, cyclic_data, transform=ccrs.PlateCarree(), cmap="plasma")
plt.colorbar(p, orientation='horizontal', pad=0.05, fraction=0.05, extend='both')

# grid lines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth = 1.5, color = 'gray', linestyle='-')
# Can only import gridline labelling for Mercator and Plate Carree

#%% Special gridlines
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
fig, ax = plt.subplots(subplot_kw=dict(projection=ccrs.PlateCarree()))
# Can use the zorder function to force features to be on top
ax.add_feature(cfeature.LAND, color='k', zorder=4)
p= ax.pcolormesh(cyclic_lons, lat, cyclic_data, transform=ccrs.PlateCarree(), cmap="plasma")
plt.colorbar(p, orientation='horizontal', pad=0.05, fraction=0.05, extend='both')

# grid lines
gl = ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth = 1.5, color = 'gray', linestyle='--')
gl.xlabels_bottom = False
gl.ylabels_left = False
gl.xformatter = LONGITUDE_FORMATTER
gl.yformatter = LATITUDE_FORMATTER

#%% 2/11/2019 MatPlotLib Tutorial Scipy lectures

# This turns on matplotlib mode which adds matplotlib integration
%matplotlib

# To have graphs show up within the console: 
%matplotlib inline

#%% Pyplot

# matplotlib is a python version of the matlab plotting procedure
# Pyplot is used primarily for this

from matplotlib import pyplot as plt
import numpy as np

#%% Simple plots

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

# This gives invalid syntax error
# This is for doing things in the ipython console
$ ipython --pylab

#%% Plotting with default settings

import numpy as np
import matplotlib.pyplot as plt

X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(S)

plt.plot(X,C)
plt.plot(X,S)
plt.show()

#%% Instantiating defaults

# Can change figure settings 

plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)

# Can change color, width, style...
plt.plot(X,C, color="blue", linewidth=1, linestyle="-")
plt.plot(X,S, color="green", linewidth=1, linestyle="-")

# Can set limits
"""I'm not retyping this section as redundant"""

# Set tick marks
plt.xticks(np.linspace(-4,4,9, endpoint=True))
plt.yticks(np.linspace(-4,4,9, endpoint=True))

# Can save figures using the function: plt.savefig("FILE NAME", dpi=##)

# To show a plot on screen
plt.show()

#%% Aesthetics some more

# Can change colors 
# The different color options are: 
# https://matplotlib.org/examples/color/named_colors.html

# Can set limits as well that would be different from the default

# Can set the spacing between tick marks

# Can set the labels for the tick marks

#%% Moving Spines

# Spines are the borders of the axes
# They can be placed at arbitrary locations
plt.figure(figsize=(8,6), dpi=80)
plt.subplot(1,1,1)
x = np.linspace(-np.pi, np.pi, 256, endpoint=True)
C, S = np.cos(X), np.sin(X)
plt.plot(X,C, color="blue", linewidth=1.5, linestyle="-", label="cosine")
plt.plot(X,S, color="coral", linewidth=1.5, linestyle="-", label="sine")

# Spines
ax = plt.gca()
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.xaxis.set_ticks_position('bottom')
ax.spines['bottom'].set_position(('data', 0))
ax.yaxis.set_ticks_position('left')
ax.spines['left'].set_position(('data',0))

# Legends
# Can set the location for a legend by using loc
# Otherwise it will default to the best location
plt.legend(loc='upper left')

# Annotating points
t = 2*np.pi/3

plt.plot([t,t], [0, np.cos(t)], color='blue', linewidth=2.5, linestyle="--")
plt.scatter([t, ], [np.cos(t), ], 50, color='blue')
plt.annotate(r'$cos(\frac{2\pi}{3})=-\frac{1}{2}$', xy=(t, np.cos(t)), xycoords='data', xytext=(-90, -50), textcoords='offset points', fontsize = 16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))

plt.plot([t,t], [0, np.sin(t)], color='red', linewidth=2.5, linestyle="--")
plt.scatter([t, ], [np.sin(t), ], 50, color='red')
plt.annotate(r'$sin(\frac{2\pi}{3})=-\frac{\sqrt{3}}{2}$', xy=(t, np.sin(t)), xycoords='data', xytext=(+10, +30), textcoords='offset points', fontsize = 16, arrowprops=dict(arrowstyle="->", connectionstyle="arc3, rad=.2"))

# The $ symbol is a special character for writing equations and things

#%% Figures and subplots

"""
Can create multplots by setting the number of rows and columns
Can change size of figure and dpi of figure
Can change axes independently for different plots in a multiplot
"""

#%% Other types of plots

"""
Many different plot types including: 
    regular (plot)
    scatter
    bar
    contour
    imshow
    quiver
    pie 
    grid
    (polar axis)
    (3D plots)
    text
"""

# Can create many different types of graphs which are useful for different things and provide different ways of visualizing different types of data

#%% 2/13/2019 Challenge Problem

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

data = xr.open_dataset("CESM.003.SST.1980.nc")
lat = np.array(data.lat)
lon = np.array(data.lon)
data = np.array(data.SST)
meandata = np.mean(data, axis = 0)

fig, ax = plt.subplots(figsize =(12,8), ncols=4, nrows=3, sharex = True, sharey = True)
#fig, ax = plt.subplots(ncols=4, nrows=3)
m1 = ax[0,0].pcolormesh(lon, lat, data[0], cmap = 'bwr')
m2 = ax[0,1].pcolormesh(lon, lat, data[1], cmap = 'bwr')
m3 = ax[0,2].pcolormesh(lon, lat, data[2], cmap = 'bwr')
m4 = ax[0,3].pcolormesh(lon, lat, data[3], cmap = 'bwr')
m5 = ax[1,0].pcolormesh(lon, lat, data[4], cmap = 'bwr')
m6 = ax[1,1].pcolormesh(lon, lat, data[5], cmap = 'bwr')
m7 = ax[1,2].pcolormesh(lon, lat, data[6], cmap = 'bwr')
m8 = ax[1,3].pcolormesh(lon, lat, data[7], cmap = 'bwr')
m9 = ax[2,0].pcolormesh(lon, lat, data[8], cmap = 'bwr')
m10 = ax[2,1].pcolormesh(lon, lat, data[9], cmap = 'bwr')
m11 = ax[2,2].pcolormesh(lon, lat, data[10], cmap = 'bwr')
m12 = ax[2,3].pcolormesh(lon, lat, data[11], cmap = 'bwr')
ax[0,0].set_title('January')
ax[0,1].set_title('February')
ax[0,2].set_title('March')
ax[0,3].set_title('April')
ax[1,0].set_title('May')
ax[1,1].set_title('June')
ax[1,2].set_title('July')
ax[1,3].set_title('August')
ax[2,0].set_title('September')
ax[2,1].set_title('October')
ax[2,2].set_title('November')
ax[2,3].set_title('December')
c1 = fig.colorbar(m1, extend="both", ax = ax)

#%% Challenge problem with a for loop

import numpy as np
import xarray as xr
import matplotlib.pyplot as plt

data = xr.open_dataset("CESM.003.SST.1980.nc")
lat = np.array(data.lat)
lon = np.array(data.lon)
data = np.array(data.SST)
months = ('January', 'February', 'March', 'April', 'May', 'June', 'July', 'August', 'September', 'October', 'November', 'December')

fig, ax = plt.subplots(figsize =(14,8), ncols=4, nrows=3, sharex = True, sharey = True)
    
n = 0 # Need to define a counter for the data and months
for x in range(0,3):
    for y in range(0,4):
        im = ax[x,y].pcolormesh(lon, lat, data[n], cmap='bwr', vmin = 0, vmax = 35)
        ax[x,y].set_title(months[n])
        n += 1 # Counter for the data referecne
# Add a title to the multiplot
fig.suptitle('Sea Surface Temperature', fontsize = 16)
# Single colorbar for the entire multiplot
plt.colorbar(im, ax = ax, extend = 'both')


#%% Test if can plot within colorbar
fig, ax = plt.subplots()
plt.colorbar(ax.pcolormesh(lon, lat, data[0]), extend='both')

#%% Lecture 5.2. Masked Arrays

%matplotlib inline
import numpy as np
import matplotlib.pyplot as plt

# Can change the default plotting parameters for matplotlib
import matplotlib as mpl
mpl.rcParams['figure.figsize'] = [6.0, 4.0]
mpl.rcParams['figure.dpi'] = 200
mpl.rcParams['savefig.dpi'] = 300
mpl.rcParams['font.size'] = 14
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['figure.titlesize'] = 'medium'
mpl.rcParams['lines.linewidth'] = 2.0

#%% Masked arrays

# array with a bad data point
x = np.array([1,2,3,-99,5])
x.mean()

mx = np.ma.masked_array(x, mask=[0,0,0,1,0])
mx
mx.mean()
mx2 = np.ma.masked_where(x<0, x)
mx2.mean()

#%% Example from tutorial

data = np.loadtxt('populations.txt')
data
data.shape
plt.plot(data[:,0], data[:,1])
plt.plot(data[:,0], data[:,2])
plt.plot(data[:,0], data[:,3])
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(['Hares', 'Lynx', 'Carrots'], loc = 'upper right')

#%% Reordering data

data.T
year, hares, lynx, carrots = data.T
print(year)

print(hares)

print(lynx)

print(carrots)

#%% Boxplots

plt.boxplot([hares, lynx, carrots], labels = ('hares', 'lynx', 'carrots'))
plt.xlabel('Species')
plt.ylabel('Population (thousands)')

#%% Problem with the data

# Mean of the whole array
data.mean()
# Mean of the first axis of the data
np.mean(data, axis=0)
data.mean(axis=0)
# Mean of specifically defined axes
print(hares.mean())
print(lynx.mean())
print(carrots.mean())

#%% Masking out bad years

# The bad years are 1903 - 1910 and 1917 - 1918
# & = and
# | = or

# Need to create a mask that encompasses only the bad data so need to create 2 ranges of data
mask = ((year >=1903) & (year<=1910)) | ((year>=1917) & (year<=1918))

lynx_masked = np.ma.masked_where(mask, lynx)
lynx_masked
hares_masked = np.ma.masked_where(mask, hares)

plt.plot(year, hares_masked)
plt.plot(year, lynx_masked)
plt.plot(year, carrots)

#%% Comparing means

print(hares.mean())
print(hares_masked.mean())
print(lynx.mean())
print(lynx_masked.mean())

#%% Boxplots of masked data

plt.boxplot([hares, hares_masked, lynx, lynx_masked, carrots], labels = ('hares', 'masked hares', 'lynx', 'masked lynx', 'carrots'))
plt.ylabel('Population (thousands)')
plt.xlabel('Species')

#%% Challenge problem 1

# Hare numbers bad over 60000

mask1 = (hares > 60000)

# Note, all the hare populations are over 6000

hare_masked1 = np.ma.masked_where(mask1, hares)
print(hare_masked1.mean())
print(hares.mean())
print(plt.plot(year, hares, linestyle = '-', linewidth=6.0), plt.plot(year, hare_masked1, color = 'red', linestyle = '--'))

#%% Boxplot

plt.boxplot([hares, hare_masked1], labels = ('hares', 'masked hares'))

#%% Challenge problem 2

p2_data = np.loadtxt('Populations.txt')
year, hares, lynx, carrots = p2_data.T

from scipy.stats.stats import pearsonr
help(pearsonr)

"""
Input = the 2 datasests to test
Output = Correlation coefficient , P-value
"""

print(pearsonr(hares, lynx))
print(pearsonr(hares, carrots))
print(pearsonr(lynx, carrots))

#%% Correlation with lag time

plt.plot(data[:,0], data[:,1])
plt.plot(data[:,0], data[:,2])
plt.plot(data[:,0], data[:,3])
plt.xlabel('Year')
plt.ylabel('Population')
plt.legend(['Hares', 'Lynx', 'Carrots'], loc = 'upper right')

#%% Correlations + lagged correlations
print("")
print("Hares and Lynx no lag = ", str(pearsonr(hares, lynx)))
print("Hares and lynx lag = ", str(pearsonr(hares[:20], lynx[1:])))
print("")
print("Carrots and Hares no lag = ", str(pearsonr(carrots, hares)))
print("Carrots and Hares lag = ", str(pearsonr(carrots[:20], hares[1:])))
print("")
print("Carrots and Lynx lag 2 years = ", str(pearsonr(carrots[:19], lynx[2:])))
print("")
print("The One year lagged correlations are much better than the not lagged correlation")
print("The Two year lagged correlation between carrots and lynx is not significant")
print("")
fig, ax = plt.subplots(ncols = 2, nrows = 2, sharey = True)
fig.subplots_adjust(hspace=0.5)  # use this to adjust spacing between subplots
ax[0,0].scatter(hares, lynx), ax[0,0].set_title('Hares and Lynx')
ax[0,1].scatter(hares[:20], lynx[1:]), ax[0,1].set_title('Lagged Hares and Lynx')
ax[1,0].scatter(carrots, hares), ax[1,0].set_title('Carrots and Hares')
ax[1,1].scatter(carrots[:20], lynx[1:]), ax[1,1].set_title('Lagged Carrots and Hares')
plt.suptitle('Correlation between populations', fontsize = 16)

#%% Alternate ways to calculate correlation

np.corrcoef(hares, lynx)
# This one does not output a p value so maybe not as useful as the above method