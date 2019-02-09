# -*- coding: utf-8 -*-
"""
Created on Sat Feb  9 13:08:02 2019

@author: tomyamashita
"""

#%% 2/4/2019 Stuff

import numpy as np

# Use mgrid to create multiple arrays with different structures
# Can assign multiple variables to the grid so output is separate in each grid
x, y = np.mgrid[0:5,0:6]
x
y

#%% Importing Files

# Not sure what this function is but it doesn't work
!head 'Mortality_Dataset_20190101.csv'

# These aren't working. Not sure what's going on
mortalities = np.genfromtxt('Mortality_Dataset_20190101.txt')
mortalities.shape

#%% 2/4/2019 Class Stuff

"""
This week's plan: 
    Monday- Numpy (Lecture + Lab)
    Wednesday- matplotlib (Riley Brady)
"""

#%% Lecture

import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline

# Arrays and aranges
a = np.arange(10)
a

plt.plot(a, 3*a+5)
plot.title("My plot")
plt.xlabel("a")
plt.ylabel("3*a+5")

#%% Arrarys

b = np.zeros((5,6))
b[0,0] = 1 # variable array[row,column]
b
b[0,1] = 3
b[1,] = 5 # Can also use b[1,:] = 5
b
plt.pcolormesh(b, cmap = 'inferno')
plt.colorbar()

#%% New array that is part of the old one
c = b[0:2,:]
c

#%% Types of arrays

# random arrays
d = np.random.rand(100)
plt.plot(d)
# Can do many plot manipulations including adding labels and titles and things
# x axis in plot is the elements and y is value
d.max()
d.min()
d.mean()
d.std()
# Does functions by using the variable.function() rather than function(variable)
# Although both methods work
max(d)

plt.hist(d)
e = np.random.rand(1000)
plt.hist(e, bins = 20)

#%% More random numbers

# Creating random numbers with a normal distribution
f = np.random.randn(1000)
plt.hist(f)

#%% Scypy lecture notes for today

# 1.3.1.6 Copies and views
a = np.arange(10)
print(a)
b = a[::2]
print(b)
print(np.may_share_memory(a,b))
b[0] = 12
a
# If you don't make a copy of the dataset, when you manipulate one, it manipulates the other

a = np.arange(10)
c = a[::2].copy() # This forces a copy to be created
c[0] = 12
a
c
np.may_share_memory(a,c)

#%% 1.3.1.7 Fancy Indexing

np.random.seed(3)  # Can be used to ensure that the same set of random numbers will be selected each time 
a = np.random.randint(0,21,15)
a
(a % 3 == 0)
mask = (a % 3 == 0)
extract_from_a = a[mask]
extract_from_a
a[a % 3 == 0] = -1 # This is saying, if an element in a is a multiple of 3, then make the value -1
a

# Can index the same value in an array multiple times

#%% 1.3.2.1 Elementwise operations

# Basic operations
a = np.array([1,2,3,4])
a + 1
2**a

# Arithmetic operates elementwise (on individual elements in the array)

b = np.ones(4) +1
a - b
a * b

j = np.arange(5)
2**(j+1) - j
# Performing operations on arrays is much faster than on ranges

# Matrix multiplication
c = np.ones((3,3))
c*c
c.dot(c)

a + (a-1)
a
a[0:4:2] + a[1:4:2]

#%% Other Operations
a = np.array([1,2,3,4])
b = np.array([4,2,2,4])
a == b  # Boolean operator on the individual elements not the array itself
a > b

# Arraywise comparisons
c = np.array([1,2,3,4])
np.array_equal(a,b)
np.array_equal(a,c)

# Logical operators
a = np.array([1,1,0,0], dtype=bool) # When type()= bool, 1 represents TRUE, 0 represents FALSE
b = np.array([1,0,1,0], dtype=bool)
np.logical_or(a,b)
np.logical_and(a,b)
np.logical_and(a,b)=False # Cannot assign a value to the logical statement

# Transcendental functions
a = np.arange(5)
np.sin(a)
np.log(a)
np.exp(a)

# Shape mismatches
a = np.arange(4)
a + np.array([1,2])
# Cannot perform operations on 2 different sized arrays

# Transposition
# Upper triangle array
a = np.triu(np.ones((3,3)), 1)
a
# To transpose rows and columns, use the "T" function
a.T

# Excercises
np.allclose # Look at help for this function using ctrl+i
# Checks if 2 arrays are equal within a tolerance range
# Could be useful if checking arrays with different numbers of decimal points
np.tril  # Look at help for this function 
# Lower triangle of an array. Basically the opposite of np.triu

#%% 1.3.2.2 Basic reductions

# Computing sums
x = np.array([1,2,3,4])
x
np.sum(x)
x.sum()

# Sum by rows and by columns
x = np.array([[1,1], [2,2]])
x
x.sum(axis=0)  # Array axes: axis 0 = columns; axis 1 = rows
x[:, 0].sum(), x[:,1].sum()
# Note: remember things are arranged as rows, columns 
x.sum(axis=1)
x[0,:].sum(), x[1,:].sum()

# Similar in higher dimensions
# axis can equal up to the number of dimensions in the array
x = np.random.rand(2,2,2)
x.sum(axis=2)[0,1]
x[0,1,:].sum()

# Other reductions
x = np.array([1,3,2])
x.min()
x.max()
x.argmin()
x.argmax()

# Logical operations
np.all([True,True,False])
np.any([True,True,False])
a = np.zeros((100,100))
np.any(a != 0)
np.all(a == a)
a = np.array([1,2,3,2])
b = np.array([2,2,3,2])
c = np.array([6,4,4,5])
# Checks to see if all things are correct. i.e. element a[0] <= b[0], etc. and element b[0] <= c[0]
((a <= b) & (b <= c)).all()

# Statistics
x = np.array([1,2,3,1])
y = np.array([[1,2,3],[5,6,1]])
x.mean()
np.median(x) # Cannot use x.median()
# When using the x.function, it is recalling attributes of the data, while function(x) is calculating the result from the data
np.median(y, axis=-1) # axis specifies the last axis (rows in this case)
x.std()

#%% Exercises

np.sum  # Help for this function
np.cumsum  # Help for this function
# Returns the cumulative sum along a given axis
# This will output individual sums in the array as it adds numbers together
x.sum()
x.cumsum()


#%%  Populations exercise
!cat # This is some function not necessary associated with python. Possibly a unix function
# Need the double backslash \\ in file path because \ is a special character in python so need second one to tell python that it is not a special character
data =  np.loadtxt('C:\\Users\\rcyam\\Dropbox\\UTRGV\\2Spring2019\\MARS5470_Intro_Sci_Programming\\Python Code\\populations.txt')
data
# If working directory is specified, you do not need the file path
data2 = np.loadtxt('populations.txt')
data2
year, hares, lynxes, carrots = data.T  # Not sure what this is doing because it didn't change anything
data

from matplotlib import pyplot as plt
plt.axes([0.2,0.1,0.5,0.8]) # Sets the dimensions of the plot as [left, bottom, width (right), height (top)] and uses normalized units ranging from 0-1
# Cannot set absolute axes, so need to know what values of the data are to normalize
plt.plot(year, hares, year, lynxes, year, carrots)
plt.legend(('Hare', 'Lynx', 'Carrot'), loc=(1.05,0.5))
# Sets the legend and the location of the plot
# Plot area is from 0:1 so anything <0 and >1 is outside the plot area

populations = data[:,1:]
populations
populations.mean(axis=0)  # Remember, axis 0 = columns
populations.std(axis=0)
np.argmax(populations, axis=1)  # Checks to see which species has the largest population in each year

#%% Diffusion Example

n_stories = 1000 # Number of walkers
t_max = 200 # number of "steps" taken
t = np.arange(t_max)
# 2 * (a random number between 0 and 1 for each value of a matrix defined by stories and time) - 1
# The multiplication will result in either 2*1 or 2*0 then the -1 makes values either -1 or 1
steps = 2 * np.random.randint(0,1+1, (n_stories, t_max)) - 1
np.unique(steps) # Checks that all values are either 1 or -1
steps
np.shape(steps)
positions = np.cumsum(steps, axis=1)  # Cumulative sum of the time dimension
sq_distance = positions ** 2
mean_sq_distance = np.mean(sq_distance, axis=0)
plt.figure(figsize=(4,3))
plt.plot(t, np.sqrt(mean_sq_distance), 'g.', t, np.sqrt(t), 'y-') # the 'g.' is a shorthand for color and style for the plot
# Plots x, y and x, sqrt(x)
plt.xlabel(r"$t$")
plt.ylabel(r"$\sqrt{\langle (\delta x)^2 \rangle}$")  # Inserts a symbol label
plt.tight_layout() # Ensures there is enough space for labels

#%% 2/6/19 Lecture Stuff. Riley Brady Guest Lecture

# Going through the powerpoint on color

import matplotlib.pyplot as plt  # plt is a standard convention for naming the matplotlib package
import numpy as np
%matplotlib inline

# Can use plt.subplots or plt.figure to specify figure size, etc.

plt.subplots(figsize=(2,2))

# Can be used to create multiplots and other complex plots
fig, ax = plt.subplots(figsize=(2,2), nrows=2, ncols=2)

#%% Line plots

x = np.linspace(-np.pi, np.pi, 64, endpoint=True)
y = np.cos(x)
z = np.sin(x)
plt.plot(y)
fig, ax = plt.subplots(figsize=(3,3))
ax.plot(x,z)
ax.plot(x,y)

fig, ax = plt.subplots(figsize=(12,6), ncols=2)
ax[0].plot(x,y)
ax[1].plot(x,z)

# Useful to google documentation for different functions

#%%
fig, ax = plt.subplots(figsize=(4,4))
ax.plot(x,y, label="cosine", linewidth=3, linestyle='-',color = "blue", marker = 'o')
ax.plot(x,z, label="sine", linewidth = 2, linestyle='-.', color = "red", marker = '*')
plt.legend()  # Defaults to the best place within the plot

"""
Different keywords for plots:
    color = color of the line. Can be in hexcodes or as color names
    linewidth = width of the line
    linestyle = style of the line plotted. There are many different styles but must be specified as a string
        '-' is just a normal line
    marker = type of marker to plot over the line
    markersize = change the size of the marker
"""

#%% Axes and labelling

"""
Can set a number of different plot properties
    xlim, ylim = limits of the axes
    title: title of the plot
    xlabel, ylabel = label for the axes
    xticks, yticks = Defines what ticks to show
    xticklabels, yticklabels = how to label the tick marks
"""

fig, ax = plt.subplots(figsize=(6,6))
ax.plot(x,y, color="blue")
ax.plot(x,z, color="red")
ax.set_xlim([-4,4])
ax.set_ylim([-2,2])
ax.set_title('Sine and Cosine', fontsize = 18)
# Can compress all the set functions into a single line

#%% Style Sheets

# Can set different styles so you can choose nice looking plots without having to type individual lines for customization

plt.style.available
plt.style.use('fivethirtyeight')
fig, ax = plt.subplots()
ax.plot(x,y)

#%% Meshes

# Can use xarray to do data import and things
import xarray as xr

#%% Here's what I did today after lecture

"""
Spent a lot of time trying to update packages and install packages that were introduced in Riley's lecture
i.e. xarray and netcdf4
"""

"""
Figured out that, on personal computers, you need to make sure that the program is installed for "only single user" 
or you need to provide administrative privileges for all users. 
On Windows 10 (as compared to Windows 7), the administrative privileges option is much more difficult
Once you have done this, package installation works as normal. 
"""

"""
I had to reinstall Anaconda on my laptop using the "only single user" option to make package installation work
I was receiving a "permission denied" error when using "for all users" option indicating a weird lack of administrative privileges
despite having only 1 account on the machine (which had administrative privileges). 
"""

"""
NOTE: on the lab computers, package installation may not work because of the administrative privileges problem
I can't remember if I tried it or not
"""
