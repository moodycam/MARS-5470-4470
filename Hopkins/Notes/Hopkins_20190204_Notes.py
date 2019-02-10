# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
"""
Board Notes 20190204
"""
#%%
import numpy as np # Imports numpy
import matplotlib.pyplot as plt # Imports matplotlib
#%%
a = np.arange(10)
#%%
plt.plot(a, 3*a +5)
plt.xlabel('a')
plt.ylabel('3a+5')
plt.title('myplot')
plt.show()
#%%
b = np.zeros((5,6))
b
#%%
b[0,0] = 10
b[0,1] = 13
b
#%%
plt.pcolormesh(b)
plt.colorbar()
#%%
b[ 1, :] = 5 # Sets everything in row '1'
plt.pcolormesh(b, cmap = 'inferno') # Specifies colors
plt.colorbar()
#%%
c = b[0:2, :] # Sets a new array based on just the first two rows of 'b'
c
plt.pcolormesh(c) # Adds color to the plot
plt.colorbar()
#%%
d = np.random.rand(1000) # Makes a completely random array
d = np.random.randn(1000) # Makes a random array that is evenly distributed
d
#%%
plt.plot(d)
plt.xlabel('index')
plt.ylabel('d')
#%%
d.max()
#%%
d.min()
#%%
d.mean()
#%%
d.std()
#%%
plt.hist(d, bins=20)
#%%
""" 'And' and 'Or' Notes """

# 'and' means both 'a' and 'b' statements need to be true. Otherwise everything
    # else is false
# 'or' means that at least one of 'a' and 'b' need to be true. Only if both
    # are false will it not work
#%%
"""
Sci-Py Numpy Notes
    :1.3.2.
"""
#%% Basic Operations
a = np.array([1,2,3,4]) # Creates an array
a + 1 # Adds '1' to every number in 'a'
2**a # Multiplies all of the values in 'a' by 2
#%% All Arithmetic Operates Elementwise
b = np.ones(4) + 1 # Creates an array of four floats that are all just '2'
b
a - b # Subtracts 'b' from 'a'
a * b # Mupltiplies 'a' and 'b'
j = np.arange(5)
2** (j + 1) - j
#%%
a = np.arange(10000)
%timeit a + 1 # Times how long it takes to do something, not super important
#%% Comparisons
a = np.array([1,2,3,4])
b = np.array([4,2,2,4])
a == b # Checks to see which values in each array are the same
a > b # Checks to see if its higher
#%% Array-Wise Comparisons
a = np.array([1,2,3,4])
b = np.array([4,2,2,4])
c = np.array([1,2,3,4])
np.array_equal(a,b) # Let's you know if the two array sare equal
#%% Logical Operations
np.array_equal(a,c) # Same as above
#%% Logical Operations
a = np.array([1,1,0,0, dtype=bool])
b = np.array([1,0,1,0, dtype=bool])
np.logical_or(a,b) # Did not work, don't know why
#%% Logical Operations
np.logical_and(a,b) # Not quite sure what this actually does at it says
    # everything is true and its not...
#%% Transcendental Functions
a = np.arange(5)
np.sin(a)
np.log(a)
np.exp(a)
#%% Shape Mismatches
a = np.arange(4)
a + np.array([1,2])
#%% Transposition
a = np.triu(np.ones((3,3)), 1)
a
a.T # Changes the view of the array using transposition
# Note: informs you to also check np.linalg
# Note: informs you to also check np.allclose
# Note: informs you to also check np.triu and np.tril
#%%
""" Computing Sums """
#%%
x = np.array([1,2,3,4])
np.sum(x) # Sums all of the values in X
x.sum() # Does the same as above
# Check saved image in Week 4 on BEANS
#%% Sum by Rows and by Columns
x = np.array([[1,1], [2,2]])
x
x.sum(axis=0) # Sums the columns in the first dimension (i.e. vertical)
x.sum(axis=1) # Sums the rows in the second dimension (i.e. horizontal)
x[0, :].sum(), x[1, :].sum() # Asks for the sum of all of the 0 dimension then
    # all of the sums in the 1 dimension
#%% Same Idea in Higher Dimensions
x = np.random.rand(2,2,2)
x.sum(axis=2)[0,1]
x[0,1,:].sum()
#%% Extrema
x = np.array([1,3,2])
x.min()
x.max()
x.argmin() # Index of minimum
x.argmax() # Index of maximum
#%% Logical Operations
np.all([True, True, False])
np.any([True, True, False])
#%% Statistics
x = np.array([1,2,3,1])
y = np.array([[1,2,3], [5,6,1]])
x.mean()
np.median(x)
np.median(y, axis=-1)
x.std()
#%%
"""
Array Shape Manipulation
"""
#%% Flattening
a = np.array([[1,2,3], [4,5,6]])
a.ravel()
a.T
a.T.ravel() # Higher dimensions: last dimensions ravel out first
#%% Reshaping
a.shape
b=a.ravel()
b=b.reshape((2,3))
b
#%% Another Way to Reshape
b[0,0] = 99
a
a = np.zeros((3,2))
b = a.T.reshape(3*2)
b[0] = 9
a
#%% Adding A Dimension
z = np.array([1,2,3])
z
z[:, np.newaxis] # Adds the adition of an axis to an array
z[np.newaxis, :]
#%% Dimension Shuffling
a = np.arange(4*3*2).reshape(4,3,2)
a.shape
a[0,2,1]
b = a.transpose(1,2,0)
b.shape
b[2,1,0]
#%% Resizing
a = np.arange(4)
a.resize((8,)) # Resizes an array. However, cannot be referred to somewhere
    # else
a
#%%
""" Sorting Data """
#%% Soring Along an Axis
a = np.array([[4,3,5], [1,2,1]])
b = np.sort(a, axis=1) # Check out 'axis' using Ctrl+i. Rather confusing.
b
#%% In-Place Sort
a.sort(axis=1)
a
#%% Sorting With Fancy Indexing
a = np.array([4,3,1,2])
j = np.argsort(a)
j
a
#%% Finding minima and maxima
a = np.array([4,3,1,2])
j_max = np.argmax(a)
j_min = np.argmin(a)
j_max, j_min

