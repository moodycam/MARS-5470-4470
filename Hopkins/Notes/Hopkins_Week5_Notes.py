# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
"""
2019-02-11
"""
#%% 
""" Inputting Data from CSV/Excel """
ghostall = 'E:\Thesis\Preliminary_Array\AllData_Stats\ghostsall_python.xlsx'
sheet = 'ghostsall'
#%%
import pandas as pd
df = pd.read_excel(io=ghostall, sheet_name=sheet)
print(df.head(5))
#%%
import numpy as np
import matplotlib.pyplot as plt
#%%
df.reset_index().values
#%%
dfa = df.values
#%%
df.reset_index().values.ravel().view(dtype=[('index', int), ('ID', int), ('Mon', int), ('Freq', int)])
#%%
plt.pcolormesh(dfa[0,:,:])
#%%
"""
2019-02-13
"""
#%%
""" Matplotlib Tutorial from Sci-Py """
#%%
# Imports
from matplotlib import pyplot as plt
import numpy as np
#%% Draw Cosine and Sine Funtions on the Same Plot
X = np.linspace(-np.pi, np.pi, 256, endpoint=True)
    # X is now a numpy array with 256 values raning from -pi to + pi (included)
C, S = np.cos(X), np.sin(X)
    # C is the cosine (256 values)
    # S is the sine (256 values)
#%%
plt.plot(X,C)
plt.plot(X,S)
plt.show()
#%%
# Create a figure of size 8x6 inches, 80 dots per inch
plt.figure(figsize=(8,6), dpi=80)
# Create a new subplot from a grid of 1x1
plt.subplot(1,1,1)
X=np.linspace(-np.pi, np.pi, 256, endpoint=True)
C,S = np.cos(X), np.sin(X)
# Plot cosine with a blue continuous line of width 1 (pixels)
plt.plot(X, C, color="blue", linewidth=1.0, linestyle="-")
# Plot sine with a green continuous line of width 1 (pixels)
plt.plot(X, S, color="green", linewidth=1.0, linestyle="-")
# Set x limits
plt.xlim(-4.0, 4.0)
# Set x ticks
plt.xticks(np.linspace(-4,4,9, endpoint=True))
# Set y limits
plt.ylim(-1.0, 1.0)
# Set y ticks
plt.yticks(np.linspace(-1,1,5, endpoint=True))
# Save figure using 72 dots per inch
    # plt.savefig("exercise_2.png", dpi=72)

# Show results on Screen
plt.show()
#%% Changing Colors and Linewidths
plt.figure(figsize=(10,6), dpi=80)
plt.plot(X, C, color='blue', linewidth=2.5, linestyle='-')
plt.plot(X,S, color="red", linewidth=2.5, linestyle="-")
#%% Setting Limits
plt.figure(figsize=(10,6), dpi=80)
plt.plot(X, C, color='blue', linewidth=2.5, linestyle='-')
plt.plot(X,S, color="red", linewidth=2.5, linestyle="-")
plt.xlim(X.min() * 1.1, X.max * 1.1)
plt.ylim(C.min() * 1.1, C.max * 1.1)
#%% Setting Ticks
plt.figure(figsize=(10,6), dpi=80)
plt.plot(X, C, color='blue', linewidth=2.5, linestyle='-')
plt.plot(X,S, color="red", linewidth=2.5, linestyle="-")
plt.xlim(X.min() * 1.1, X.max * 1.1)
plt.ylim(C.min() * 1.1, C.max * 1.1)
plt.xticks([-np.pi, -np.pi/2, 0, np.pi/2, np.pi])
plt.yticks([-1, 0, +1])
#%%
"""
Constructed Graphs for Another Class
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import xarray as xr
#%%
data = np.loadtxt('C:\\Users\Miles\Downloads\populations.txt')
#%%
data
data.T
year, hares, lynxes, carrots = data.T
#%%
year
#%%
hares
#%%
lynxes
#%%
print(hares.mean())
print(lynxes.mean())
print(carrots.mean())
#%% Figure 1 (Line)
plt.subplots(figsize=(17,10))
plt.plot(data[:,0], data[:,3], color="black", linewidth=5)
plt.plot(data[:,0], data[:,1], color="midnightblue", linewidth=5)
plt.plot(data[:,0], data[:,2], color="red", linewidth=5)
plt.xlabel('Year', fontsize=24)
plt.ylabel('Population', fontsize=24)
plt.xticks(np.linspace(1900,1920,5, endpoint=True), fontsize=20)
plt.yticks(np.linspace(0,80000,9, endpoint=True), fontsize=20)
plt.legend(['Wild Carrot', 'Hare', 'Lynx'], fontsize=24, loc = 'upper right')
plt.title('Populations of Hare, Lynx, & Wild Carrot, 1900 to 1920', fontsize=34)
#%% Figure 2 (Bar and Line)
plt.subplots(figsize=(17,10))
plt.plot(data[:,0], data[:,3], color="black", linewidth=3)
plt.bar(data[:,0], data[:,1], color="midnightblue")
plt.bar(data[:,0], data[:,2], color="red", alpha=0.7)
plt.xlabel('Year', fontsize=24)
plt.ylabel('Population', fontsize=24)
plt.xticks(np.linspace(1900,1920,5, endpoint=True), fontsize=20)
plt.yticks(np.linspace(0,80000,9, endpoint=True), fontsize=20)
plt.legend(['Wild Carrot', 'Hare', 'Lynx'], fontsize=24, loc = 'upper right')
plt.title('Populations of Hare, Lynx, & Wild Carrot, 1900 to 1920', fontsize=34)
#%% Figure 3 (Just Bar)
plt.subplots(figsize=(17,10))
plt.bar(data[:,0], data[:,1], color="midnightblue")
plt.bar(data[:,0], -data[:,2], color="red")
plt.xlabel('Year', fontsize=24)
plt.ylabel('Population (Thousands)', fontsize=24)
plt.xticks(np.linspace(1900,1920,5, endpoint=True), fontsize=20)
plt.yticks([-80000, -60000, -40000, -20000, 0, 20000, 40000, 60000, 80000], [r'$80$', r'$60$', r'$40$', r'$20$', r'$0$', r'$20$', r'$40$', r'$60$', r'$80$'], fontsize=20)
plt.legend(['Hare', 'Lynx'], fontsize=24, loc = 'upper right')
plt.title('Populations of Hare & Lynx, 1900 to 1920', fontsize=34)
