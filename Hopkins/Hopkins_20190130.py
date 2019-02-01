# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
import numpy as np
#%%
"""
In Class Assignment
    Exercise: Simple Arrays
"""

"""
Exercise 1
    Create a simple two dimensional array. First, redo the examples from above.
    and then create your own: how about odd numbers counting backwards
    on the first row, and even numbers on the second?
"""
a = np.array([[0,1,2], [3,4,5]])
a
b = np.array([[5,3,1], [2,4,6]])
b
"""
Exercise 2
    Use the functions len(), numpy.shape() on these arrays. How do they relate
    to each other? And to the ndim attribute of the arrays?
"""
len(b)
np.shape(b)
b.ndim
#%%
L = range(1000)
%timeit [i**2 for i in L]
#%%
a = np.arange(1000)
%timeit a**2
#%%
np.array?
np.lookfor('create array')
# Can look up functions by typing in 'np.' and it will provide a list of
    # possible choices that will autofill. Allows you to find what you want
    # to use without having to recall the exact syntax.
#%%
"""1-D Array"""
#%%
a = np.array([0, 1, 2, 3])
a
a.ndim
a.shape
len(a)

#%%
"""2- and 3-D Arrays"""
#%%
b = np.array([[0, 1, 2], [3, 4, 5]])    # Creates 2 x 3 array
b
b.ndim # Provides the number of rows
b.shape # Provides the number of rows and then columns
len(b) # Provides the number of rows
#%%
c = np.array ([[[1], [2]], [[3], [4]]]) # Creates a 1 x 4 array
c
c.shape

#%%
"""Functions for Creating Arrays"""
#%%
a = np.arange(10) # Creates an arrange of 0 to 9
a
#%%
b = np.arange(1, 9, 2) # Start, end (exclusive), step
b

#%%
"""
Assignment: Make an Array of Even Integers Up to 30
"""

upto = np.arange(2, 31, 2)
upto
#%%
"""
Assignment: Make an Array of Negative Integers from -2 to -30
"""

dnto =  np.arange(-2, -31, -2)
dnto

#%%
"""By Number of Points"""
#%
c = np.linspace(0, 1, 6) # Start, end, num-points
c
#%%
d = np.linspace(0,1,5, endpoint = False)
d

#%%
"""Common Arrays"""
#%%
a = np.ones((3,3))
a
#%%
b = np.zeros((2,2))
b
#%%
c = np.eye(3)
c
#%%
d = np.diag(np.array([1,2,3,4]))
d

#%%
"""Random Numbers"""
#%%
a = np.random.rand(4)
a
#%%
b = np.random.randn(4)
b
#%%
np.random.seed(1234)

#%%
"""
Exercise: Creating Arrays Using Functions
"""
# Arange
give = np.arange(1, 100, 2)
give
#%%
# Linspace
lin = np.linspace(0,2,6)
lin
#%%
np.empty?

#%%
"""Basic Data Types"""
a = np.array([1,2,3])
a.dtype
#%%
b = np.array([1., 2., 3.,])
    # If you wnat to make it a float instead of an integer just place a '.'
        # behind the number to automatically make it a float
b.dtype

#%%
"""Basic Visualization"""
%matplotlib inline
import matplotlib.pyplot as plt
#%%
plt.plot(x,y)   # line plot
plt.show # <-- shows the plot
#%%
"""1-D Plotting"""
x = np.linspace(0, 3, 20)
y = np.linspace(0,9,20)
plt.plot(x,y) # Line plot
plt.plot(x, y, 'o') # Dot plot

#%%
"""2-D Plotting"""
image = np.random.rand(30,30)
image.shape
plt.imshow(image, cmap=plt.cm.hot)
plt.colorbar()
#%%
"""Another 2-D Plotting Option"""
plt.pcolormesh(image)
plt.colorbar()
#%%
"""
Exercise: Simple Visualizations
"""
y = np.cos(x)
plt.pcolormesh(image, cmap = gray)
!!!!! DID NOT COMPLETE !!!!
#%%%
"""Indexing and Slicing"""
a = np.arange(10)
a
a[0], a[2], a[-1]
#%%
a[::-1]
#%%
a = np.diag(np.arange(3))
a
a[1,1] # Provies the middle value
a[2,1] = 10 # Changes the third line and second row value to 10
a
a[1] # Provides the second row
## In 2D, the first dimension corresponds to rows, the second to columns
## For multidimensional plots (say 'a'), a[0] is interpreted by taking all
    ## elements in the usnpecified dimensions.
#%%
a = np.arange(10)
a
a[2:9:3]
a[:4] # Note last index is not included
## By default, start is 0, end is the last step, and step is 1
#%%
a[1:3]
a[::2]
a[3:]
#%%
a = np.arange(10)
a[5:] = 10
a
b = np.arange(5)
a[5:] = b[::-1]
a
#%%

