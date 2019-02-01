#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 13:30:54 2019

@author: roryeggleston
"""

import numpy as Numpy
#%%
a = Numpy.array([0, 1, 2, 3])
a
#%%
a = Numpy.array([0, 1, 2, 3])
#%%
a.ndim
#%%
a.shape
#%%
len(a)
#%%
b = Numpy.array([[0, 1, 2], [3, 4, 5]])
b
#%%
b.ndim
#%%
b.shape
#%%
len(b)
#%%
c = Numpy.array([[[1], [2]], [[3],[4]]])
c
#%%
c.shape
#%%
d = Numpy.array([[7, 5, 3, 1], [8, 6, 4, 2]])
d
#%%
d.shape
#%%
len(d)
#%%
d.ndim
#%%
a = Numpy.arange(10)
a
#%%
b = Numpy.arange(1, 9, 2)
b
#%%
c = Numpy.linspace(0,1,6)
#(start, stop, number of points)
c
#%%
#make an array that has even integers up to 30 using the above method
d = Numpy.arange(2, 32, 2)
d
#%%
e = Numpy.arange(-2,-32, -2)
#(start, stop, +stride)
e
#%%
d = Numpy.linspace(0,1,5,endpoint=False)
d
#%%
a = Numpy.zeros([3,5])
a
#%%
#change the 3rd row, 4th column element to 3
a[2,3] = 3
a
#%%
#random numbers between 0 an 1
a = Numpy.random.rand(4)
a
#%%
b = Numpy.random.randn(4)
b
b * 20
#%%
Numpy.random.seed(1234)
#%%
a = Numpy.arange(1,45,3)
a
#%%
b = Numpy.linspace(0, 2, 7)
b
#%%
c = Numpy.ones([3,6])
c
#%%
d = Numpy.zeros([45, 23])
d
#%%
e = Numpy.random.rand(15)
e
#%%
f = Numpy.random.randn(23)
f
#%%
Numpy.random.seed(16)
a = Numpy.random.rand(6)
a
#%%
Numpy.empty?
#%%
a = Numpy.empty([3,4], dtype = int)
a
#%%
a = Numpy.array([1,2,3])
a.dtype
#%%
b = Numpy.array([1.,2.,3.])
b.dtype
#%%
c = Numpy.array([1,2,3], dtype=float)
c.dtype
#%%
a = Numpy.ones((3,3))
a.dtype
#%%
d = Numpy.array([1+2j, 3+4j, 5+6j])
d.dtype
#%%
%matplotlib inline
#%%
x = Numpy.linspace(0,3,20)
y = Numpy.linspace(0,9,20)
plt.plot(x,y)
plt.plot(x,y, 'o')
#%%
x = Numpy.linspace(0,3,20)
y = Numpy.linspace(0,9,20)
plt.plot(x,y)
plt.plot(x,y, 'o')
#%%
image = Numpy.random.rand(30,30)
plt.imshow(image,cmap=plt.cm.hot)
plt.colorbar()
#%%
y = Numpy.sin(x)
plt.plot(y)
#%%
y = Numpy.cos(x)
plt.plot(x,y)
plt.xlabel('time')
plt.ylabel('cosx')
#%%
image = Numpy.random.rand(20,20)
plt.imshow(image,cmap=plt.cm.jet)
plt.colorbar()
#%%
a = Numpy.arange(10)
a
#%%
a[0], a[2], a[-1]
#%%
a[::-1]
#%%
a = Numpy.diag(Numpy.arange(3))
a
#%%
a[1,1]
#%%
a[2,1] = 10
a
#%%
a[1]
#%%
a = Numpy.arange(10)
a
#%%
a[2:9:3]
#%%
a[:4]
#%%
a[1:3]
#%%
a[::2]
#%%
a[3:]
#%%
a = Numpy.arange(10)
a[5:] = 10
a
#%%
b = Numpy.arange(5)
a[5:] = b[::b-1]
a
#%%
a = Numpy.arange(35)
a
#%%
a[1:31:4]
#%%
b = Numpy.linspace(0,120,120,dtype='int')
b
#%%
b[120:1:-2]
#%%
Numpy.arange(6) + Numpy.arange(0, 51, 10)[:, Numpy.newaxis]
#%%
a = Numpy.array([[1,1,1,1],[1,1,1,1],[1,1,1,2],[1,6,1,1]])
a
#%%
b = Numpy.array([[0.,0.,0.,0.,0.],[2.,0.,0.,0.,0.],[0.,3.,0.,0.,0.],[0.,0.,4.,0.,0.],[0.,0.,0.,5.,0.],[0.,0.,0.,0.,6.]])
b
#%%
a = Numpy.arange(7,dtype='float')
a
#%%
b = Numpy.diag(a)
b
#%%
Numpy.tile?
#%%
a= Numpy.array([[4,3], [2,1]])
a
Numpy.tile(a, (2,3))
#%%
