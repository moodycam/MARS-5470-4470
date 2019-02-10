#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Feb  4 13:50:49 2019

@author: roryeggleston
"""

import numpy as np
#%%
a = np.array([1, 2, 3, 4])
a + 1
2 ** a
#%%
import matplotlib.pyplot as plt
#%%
a = np.arange(10)
a
#%%
plt.plot(a, 3*a + 5)
plt.xlabel('a')
plt.ylabel('3a+5')
plt.title('my plot')
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
b[1 , :] = 5
b
#%%
plt.pcolormesh(b, cmap = 'inferno')
plt.colorbar()
#%%
#new array that is part of the old one
c = b[0:2,:]
c
#%%
plt.pcolormesh(c)
plt.colorbar()
#%%
#randn gives you normal distrib
d = np.random.randn(1000)
d
#%%
plt.plot(d)
plt.xlabel('index')
plt.ylabel('d')
#%%
d.max()
d.min()
d.mean()
d.std()
#%%
plt.hist(d, bins = 20)
#%%
help(plt.hist)
#%%
#1.3.1.6 scipy onwards
a = np.arange(10)
a
#%%
b = a[::2]
b
#%%
np.may_share_memory(a, b)
#%%
b[0] = 12
b
#%%
a
#a now includes the changes made to b
#%%
a = np.arange(10)
c = a[::2].copy()
c[0] = 12
a
#%%
np.may_share_memory(a, c)
#%%
is_prime = np.ones((100,), dtype = bool)
#%%
is_prime[:2] = 0
is_prime
#%%
N_max = int(np.sqrt(len(is_prime) - 1))
for j in range(2, N_max + 1):
    is_prime[2*j::j] = False
N_max
#%%
print(is_prime)
#%%
np.random.seed(3)
a = np.random.randint(0, 21, 15)
a
#%%
(a % 3 == 0)
#%%
mask = (a % 3 == 0)
extract_from_a = a[mask]
extract_from_a
#%%
a[a % 3 == 0] = -1
a
#%%
a = np.arange(0, 100, 10)
a
#%%
a[[2, 3, 2, 4, 2]]
#%%
a[[9, 7]] = -100
a
#%%
a = np.arange(10)
idx = np.array([[3, 4], [9, 7]])
idx.shape
#%%
a[idx]
#%%
a = np.array([1, 2, 3, 4])
a + 1
#%%
2 ** a
#%%
b = np.ones(4) + 1
a - b
#%%
a * b
#%%
j = np.arange(5)
2**(j + 1) - j
#%%
a = np.arange(10000)
%timeit a + 1
#%%
l = range(10000)
%timeit [i+1 for i in l]
#%%
c = np.ones((3, 3))
c * c
#%%
c.dot(c)
#%%
a = np.arange(100)
a
#%%
#adds even to odd numbers in the range?
a[0:99] + a[1:100]
#%%
b = np.arange(5)
b
#%%
2**b
#%%
a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
a == b
#%%
a > b
#%%
a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
c = np.array([1, 2, 3, 4])
np.array_equal(a, b)
#the arrays are the not the same and therefore are not equal
#%%
np.array_equal(a, c)
#the arrays are the same and are therefore equal
#%%
a = np.array([1, 1, 0, 0], dtype=bool)
b = np.array([1, 0, 1, 0], dtype=bool)
np.logical_or(a, b)
#%%
np.logical_and(a, b)
#%%
a = np.arange(5)
np.sin(a)
#%%
np.log(a)
#%%
np.exp(a)
#%%
a = np.arange(4)
a + np.array([1, 2])
#%%
a = np.triu(np.ones((3, 3)), 1)
a
#%%
x = np.array([1, 2, 3, 4])
np.sum(x)
#%%
x.sum()
#%%
x = np.array([[1, 1], [2, 2]])
x
#%%
x.sum(axis=0)
#%%
x[:, 0].sum(), x[:, 1].sum()
#%%
x.sum(axis=1)
#%%
x[0, :].sum(), x[1, :].sum()
#%%
x = np.random.rand(2, 2, 2)
x.sum(axis=2)[0, 1]
#%%
x[0, 1, :].sum()
#%%
x = np.array([1, 3, 2])
x.min()
#%%
x.max()
#%%
x.argmin()
#index(location) of minimum value of the array
#%%
x.argmax()
#index(location) of maximum value of the array
#%%
np.all([True, True, False])
#%%
np.any([True, True, False])
#%%
a = np.zeros((100, 100))
np.any(a !=0)
#%%
np.all(a == a)
#%%
a = np.array([1, 2, 3, 2])
b = np.array([2, 2, 3, 2])
c = np.array([6, 4, 4, 5])
((a <= b) & (b <= c)).all()
#%%
x = np.array([1, 2, 3, 1])
y = np.array([[1, 2, 3], [5, 6, 1]])
x.mean()
#%%
np.median(x)
#%%
np.median(y, axis=-1)
#%%
x.std()
#%%
x = np.array([1, 12, 34, 54, 68, 86])
x.cumsum()
#cumulatively sums each element in the array
#%%
#EXAMPLE FOR OPENING TXT FILES IN SPYDER + BASIC STATS HOWEVER THIS WILL BE A HEADACHE W BABBLE DATA
infile = open("/Users/roryeggleston/Documents/populations.txt")
#%%
data = np.loadtxt("/Users/roryeggleston/Documents/populations.txt")
year, hares, lynxes, carrots = data.T
# assigning columns to variables
#%%
from matplotlib import pyplot as plt
plt.axes([0.2, 0.1, 0.5, 0.8])
plt.plot(year, hares, year, lynxes, year, carrots)
plt.legend(('Hare', 'Lynx', 'Carrot'), loc=(1.05, 0.5))
#%%
populations = data[:, 1:]
populations.mean(axis=0)
#%%
populations.std(axis=0)
#%%
np.argmax(populations, axis=1)
#%%
a = np.tile(np.arange(0, 40, 10), (3, 1)).T
a
#%%
b = np.array([0, 1, 2])
a + b
#%%
a = np.ones((4, 5))
a[0] = 2
a
#%%
a = np.arange(0, 40, 10)
a.shape
#%%
a = a[:, np.newaxis]
a.shape
#%%
a
#%%
a + b
#%%
mileposts = np.array([0, 198, 303, 736, 871, 1175, 1475, 1544, 1913, 2448])
distance_array = np.abs(mileposts - mileposts[:, np.newaxis])
distance_array
#%%
x, y = np.arange(5), np.arange(5)[:, np.newaxis]
distance = np.sqrt(x ** 2 + y ** 2)
distance
#%%
plt.pcolor(distance)
plt.colorbar()
#%%
x, y = np.ogrid[0:5, 0:5]
x, y
#%%
x.shape, y.shape
#%%
distance = np.sqrt(x ** 2 + y ** 2)
#%%
x, y = np.mgrid[0:4, 0:4]
x
y
#%%
a = np.array([[1, 2, 3], [4, 5, 6]])
a.ravel()
#%%
a.T
#%%
a.T.ravel()
#%%
a.shape
#%%
b = a.ravel()
b = b.reshape((2, 3))
b
#%%
a.reshape((2, -1))
#LEFT OFF AT SECOND HALF OF RESHAPING IN 1.3.2.4
#%%
b[0, 0] = 99
a
#%%
a = np.zeros((3, 2))
b = a.T.reshape(3*2)
b[0] = 9
a
#%%
z = np.array([1, 2, 3])
z
#%%
z[:, np.newaxis]
#%%
z[np.newaxis, :]
#%%
a = np.arange(4*3*2).reshape(4, 3, 2)
a.shape
#%%
a[0, 2, 1]
#%%
b = a.transpose(1, 2, 0)
b.shape
#%%
b[2, 1, 0]
#%%
b[2, 1, 0] = -1
a[0, 2, 1]
#%%
a = np.arange(4)
a.resize((8,))
a
#%%
b = a
a.resize((4,))
#%%
a = np.array([[4, 3, 5], [1, 2, 1]])
b = np.sort(a, axis = 1)
b
#%%
a.sort(axis = 1)
a
#%%
a = np.array([4, 3, 1, 2])
j = np.argsort(a)
j
#%%
a[j]
#%%
a = np.array([4, 3, 1, 2])
j_max = np.argmax(a)
j_min = np.argmin(a)
j_max, j_min