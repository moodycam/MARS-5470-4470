# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:55:22 2019

@author: srv_veralab
"""

import numpy as np
#%%
a= np.array([0, 1, 2, 3])

#This is how you define an array
#%%

b = np.array([[0, 1, 2], [3, 4, 5]])

#%%
b.ndim
# This function asks how many dimesions the array has.

#%%

b.shape

#this is array has 2 rows and 3 columns
#shape[0] for the first dimension
#shape[1] for the second dimension

#%%

len(b)

#this is the size of the first dimension

#%%

a= np.arange (10) #0 .. n-1 (!)
a
#%%

b=np.arange(1,9,2)
b

#The arrange function tells
#Start at the number 1


#%%

#Make an array that has even integers up to 30.

c=np.arange(2,32,2)
c
#%%

d=np.arange(-2,-32,-2)
d

# positive 2 doesn't work here

#%%

a= np.array ([1,2,3])
a 

#%%

a=np.random.rand(4)
a
#%%

#If you want to have random numbers between 0 and 20

a*20

#%%

#If you want to have random numbers between -10 and 10

a*20 -10

#%%

#Exercise: Creating arrays using functions

   
#    Experiment with arange, linspace, ones, zeros, eye and diag.
#    Create different kinds of arrays with random numbers.
#    Try setting the seed before creating an array with random values.
#    Look at the function np.empty. What does it do? When might this be useful?

np.empty(2,float,'C')

#Why is it filling in numbers if I put the empty function? 

#%%

a=np.ones((3,3))
a

#%%

import matplotlib.pyplot as plt


x=np.linspace(0,3,20)

plt.plot(x,np.sin(x))

#%%

x=np.linspace(0,3,20)
y=np.linspace(0,9,20)
plt.plot(x,y)

# There is a point at (0,0) and a point at (3,9). 
# Three are 20 random points that connect those two points

#%%


image=np.random.rand(30,30)
plt.imshow(image, cmap=plt.cm.hot)
plt.colorbar()

#np.random.rand(30,30) creates a 30x30 array.
#the array is filled with 90 random numbers between 0-1.
#Those numbers are assigned to a color on a color bar.

#%%

image=np.random.rand(30,30)
plt.imshow(image, cmap=plt.cm.hot)
plt.colorbar()

#%%
#These are the same but with different syntax
print(a[0::1])
print(a[::1])

#%%

a= np.arange(10)
a

a[0],a[2],a[-1]

#put -1 if you want to get the last number in the array.

#%%

a[::-1]

#%%

a=np.diag(np.arange(3))
a

#%%

#For multidimensional arrays, indexes are tuples of integers:

a=np.diag(np.arange(3))
a

a[1,1]

#The result of a[1,1] is 1 because 
#The 0 is the first column and 0 is the first row

a[2,1]=10
a
#Putting that equals sign makes it so that
#the element in the third row, second column is 10

a[1]

#this gives us the 2nd row of the array.


#%%

a=np.arange(10)
a

#%%
a[2:9:3]

































