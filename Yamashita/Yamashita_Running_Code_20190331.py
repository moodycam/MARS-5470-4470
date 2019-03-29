# -*- coding: utf-8 -*-
"""
Created on Thu Mar 28 22:44:23 2019

@author: tomyamashita
"""

#%% 3/25/2019. Lecture 10.1 Predator-Prey Models

"""
Population growth. Normally exponential
Lotka-Volterra Equations
Steady state solutions
"""

#%% Tutorial

import numpy as np
import matplotlib.pyplot as plt

alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.

#%% Define time stepping function

def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    
    return u + dt * f(u)

#%% Define function for Lotka-Volterra equations
    
def f(u):
    """Returns the rate of change of species numbers.
    
    Parameters
    ----------
    u : array of float
        array containing the solution at time n.
        
    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """
    x = u[0]
    y = u[1]
    return np.array([x*(alpha - beta*y), -y*(gamma - delta*x)])

#%% Using the equations
    
T = 15.0            # Final time
dt = 0.01           # Time increment
N = int(T/dt) + 1   # Number of time steps = Final time / time increment + 1 as an integer
x0 = 10             # Initial x value (prey?)
y0 = 2              # Initial y value (predator?)
t0 = 5              # Initial time (Not in any of the equations?)
# The t0 is a seemingly useless variable because it should be defined as a relative start time

# Initial conditions
u_euler = np.empty((N,2))  # Create an empty array with N number of rows and 2 columns

# Array containing the solution
u_euler[0] = np.array([x0,y0])  # Defines the first row as an array = to x0 and y0

# Use for loop to solve for number of each species in each time step
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)

#%% Plotting the function
    
time = np.linspace(t0, T+t0, N)  # Create array from initial time to end time with the number of time increments
x_euler = u_euler[:, 0]
y_euler = u_euler[:, 1]

plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Predator prey model')

#%% System behavior

plt.plot(x_euler, y_euler, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')

#%% Exercise 1

x = gamma/beta
y = alpha/beta

time = np.linspace(t0, T+t0, N)
x_steady = np.linspace(x, x, N)
y_steady = np.linspace(y, y, N)

plt.plot(time, x_steady, label = "prey")
plt.plot(time, y_steady, label = "predator")
plt.legend(loc='middle left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Steady state populations')

#%% phase space plot for exercise 1

plt.plot(x_steady, y_steady, '-->', markevery=5, label = 'Phase Plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')

# If steady state, then the plot will not move

#%% Exercise 1.5

alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
T = 15.0            
dt = 0.01
N = int(T/dt) + 1 
x0 = gamma/beta
y0 = alpha/beta
t0 = 0.     

u_ss = np.empty((N,2))
u_ss[0] = np.array([x0,y0])

for n in range(N-1):
    u_ss[n+1] = euler_step(u_ss[n], f, dt)

time = np.linspace(t0, T+t0, N)  
x_ss = u_ss[:, 0]
y_ss = u_ss[:, 1]

#%% Plots for Exercise 1.5

plt.plot(time, x_ss, label = "prey")
plt.plot(time, y_ss, label = "predator")
plt.legend(loc='middle left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Steady state populations')

#%% Phase plot

plt.plot(x_ss, y_ss, '-->', markevery=5, label = 'Phase Plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')

#%% Exercise 2

alpha = 3   # Prey growth rate?     (Original = 1)
beta = 1.8  # Predation Rate        (Original = 1.2)
gamma = 2   # Predator death rate?  (Original = 4)
delta = 2   # Predator growth rate? (Original = 1)
T = 15.0            
dt = 0.01
N = int(T/dt) + 1 
x0 = 10
y0 = 2
t0 = 0.     

u_modify = np.empty((N,2))
u_modify[0] = np.array([x0,y0])

for n in range(N-1):
    u_modify[n+1] = euler_step(u_modify[n], f, dt)

time = np.linspace(t0, T+t0, N)  
x_modify = u_modify[:, 0]
y_modify = u_modify[:, 1]

#%% Plots
    
plt.plot(time, x_modify, label = "prey")
plt.plot(time, y_modify, label = "predator")
plt.legend(loc='middle left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Steady state populations')

#%% Phase plot

plt.plot(x_modify, y_modify, '-->', markevery=5, label = 'Phase Plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')

#%% Exercise 3

alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
T = 15.0            
dt = 0.01
N = int(T/dt) + 1 
x0 = 10.
y0 = 2.
t0 = 0.     

def RK4(u,f,dt):
    # Runge Kutta 4th order method
    """Returns the solution at the next time-step using Runge Kutta fourth order (RK4) method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    #calculate slopes
    k1 = f(u)
    u1 = u + (dt/2.)*k1
    k2 = f(u1)
    u2 = u + (dt/2.)*k2
    k3 = f(u2)
    u3 = u + dt*k3
    k4 = f(u3)
    return u + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)

u_RK = np.empty((N,2))
u_RK[0] = np.array([x0,y0])

for n in range(N-1):
    u_RK[n+1] = RK4(u_RK[n], f, dt)

time = np.linspace(t0, T+t0, N)  
x_RK = u_RK[:, 0]
y_RK = u_RK[:, 1]

#%% Plots
    
plt.plot(time, x_RK, label = "prey")
plt.plot(time, y_RK, label = "predator")
plt.legend(loc='middle left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Steady state populations')

#%% Phase plot

plt.plot(x_RK, y_RK, '-->', markevery=5, label = 'Phase Plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')

"""
The graphs are basically the same except that it does not randomly increase the peaks over time
The phase plot is the cycle of life
"""

#%% Exercise 4

"""
Copying the zombie apocalypse code
    S: the number of susceptible victims
    Z: the number of zombies
    R: the number of people "killed"
    P: the population birth rate
    d: the chance of a natural death
    B: the chance the "zombie disease" is transmitted (an alive person becomes a zombie)
    G: the chance a dead person is resurrected into a zombie
    A: the chance a zombie is totally destroyed
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import odeint

P = 0       # birth rate
d = 0.0001  # natural death percent (per day)
B = 0.0095  # transmission percent  (per day)
G = 0.0001  # resurect percent (per day)
A = 0.0001  # destroy percent  (per day)

# solve the system dy/dt = f(y, t)
def f(y, t):
     Si = y[0]  # Population at time i?
     Zi = y[1]  # Number of zombies at time i?
     Ri = y[2]  # Number of dead people at time i?
     # the model equations (see Munz et al. 2009)
     f0 = P*Si - B*Si*Zi - d*Si          # Something related to population growth rate
     f1 = B*Si*Zi + G*Ri - A*Si*Zi    # Something Zombie growth rate
     f2 = d*Si + A*Si*Zi - G*Ri       # Total death rate for both humans and zombies
     return [f0, f1, f2]

#%% initial conditions
S0 = 500.              # initial population
Z0 = 0                 # initial zombie population
R0 = 0                 # initial death population
y0 = [S0, Z0, R0]     # initial condition vector
t  = np.linspace(0, 5., 1000)         # time grid

# solve the Differential equations
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]

# plot results
plt.figure()
plt.plot(t, S, label='Living')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Days from outbreak')
plt.ylabel('Population')
plt.title('Zombie Apocalypse - No Init. Dead Pop.; No New Births.')
plt.legend(loc=0)

#%% change the initial conditions
R0 = 0.01*S0   # 1% of initial pop is dead
y0 = [S0, Z0, R0]

# solve the DEs
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]

plt.figure()
plt.plot(t, S, label='Living')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Days from outbreak')
plt.ylabel('Population')
plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; No New Births.')
plt.legend(loc=0)

#%% change the initial conditions
R0 = 0.01*S0   # 1% of initial pop is dead
P  = 10        # 10 new births daily
y0 = [S0, Z0, R0]

# solve the DEs
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]

plt.figure()
plt.plot(t, S, label='Living')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Days from outbreak')
plt.ylabel('Population')
plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; 10 Daily Births')
plt.legend(loc=0)

#%% Custom change scenarios

"""
There is seemingly no way for the population to decline steadily and death and zombies increase steadily
"""

P = 0.001       # birth rate
d = 0.001  # natural death percent (per day)
B = 0.000001  # transmission percent  (per day)
G = 0.000001  # resurect percent (per day)
A = 0.001  # destroy percent  (per day)
S0 = 500000.                   # initial population
Z0 = 0                      # initial zombie population
R0 = 0.0                 # initial death population
y0 = [S0, Z0, R0]           # initial condition vector
t  = np.linspace(0, 500., 100000)         # time grid

def f(y, t):
     Si = y[0]  # Population at time i?
     Zi = y[1]  # Number of zombies at time i?
     Ri = y[2]  # Number of dead people at time i?
     # the model equations (see Munz et al. 2009)
     f0 = P*Si - B*Si*Zi - d*Si          # Something related to population growth rate
     f1 = B*Si*Zi + G*Ri - A*Zi    # Something Zombie growth rate
     f2 = d*Si + A*Zi - G*Ri       # Total death rate for both humans and zombies
     return [f0, f1, f2]

# solve the DEs
soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]

plt.figure()
plt.plot(t, S, label='Living')
plt.plot(t, Z, label='Zombies')
plt.plot(t, R, label='Dead')
plt.xlabel('Days from outbreak')
plt.ylabel('Population')
plt.title('Zombie Apocalypse - 1% Init. Pop. is Dead; 10 Daily Births')
plt.legend(loc=0)

#%% Phase space plots

plt.plot(S, Z, '-->', markevery=5, label = 'Phase Plot')

#%% 3/27/2019 10.2 Warmup exercise

import numpy as np
import matplotlib.pyplot as plt

#%% timestep 10 times longer

T = 15.0            # Final time
dt = 0.1           # Time increment
N = int((T/dt)) + 1   # Number of time steps = Final time / time increment + 1 as an integer
x0 = 10             # Initial x value (prey?)
y0 = 2              # Initial y value (predator?)
t0 = 0.0              # Initial time (Not in any of the equations?)
alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
# The t0 is a seemingly useless variable because it should be defined as a relative start time

#%% Define time stepping function

def euler_step(u, f, dt):
    """Returns the solution at the next time-step using Euler's method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    
    return u + dt * f(u)

#%% Define function for Lotka-Volterra equations
    
def f(u):
    """Returns the rate of change of species numbers.
    
    Parameters
    ----------
    u : array of float
        array containing the solution at time n.
        
    Returns
    -------
    dudt : array of float
        array containing the RHS given u.
    """
    x = u[0]
    y = u[1]
    return np.array([x*(alpha - beta*y), -y*(gamma - delta*x)])

#%% Initial conditions
u_euler = np.empty((N,2))  # Create an empty array with N number of rows and 2 columns

# Array containing the solution
u_euler[0] = np.array([x0,y0])  # Defines the first row as an array = to x0 and y0

# Use for loop to solve for number of each species in each time step
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
    
time = np.linspace(t0, T+t0, N)  # Create array from initial time to end time with the number of time increments
x_euler = u_euler[:, 0]
y_euler = u_euler[:, 1]

plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Predator prey model')

#%% System behavior

plt.plot(x_euler, y_euler, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')
    
#%% RK method
alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
T = 15.0            
dt = 0.1
N = int(T/dt) + 1 
x0 = 10.
y0 = 2.
t0 = 0.     

def RK4(u,f,dt):
    # Runge Kutta 4th order method
    """Returns the solution at the next time-step using Runge Kutta fourth order (RK4) method.
    
    Parameters
    ----------
    u : array of float
        solution at the previous time-step.
    f : function
        function to compute the right hand-side of the system of equation.
    dt : float
        time-increment.
    
    Returns
    -------
    u_n_plus_1 : array of float
        approximate solution at the next time step.
    """
    #calculate slopes
    k1 = f(u)
    u1 = u + (dt/2.)*k1
    k2 = f(u1)
    u2 = u + (dt/2.)*k2
    k3 = f(u2)
    u3 = u + dt*k3
    k4 = f(u3)
    return u + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)

u_RK = np.empty((N,2))
u_RK[0] = np.array([x0,y0])

for n in range(N-1):
    u_RK[n+1] = RK4(u_RK[n], f, dt)

time = np.linspace(t0, T+t0, N)  
x_RK = u_RK[:, 0]
y_RK = u_RK[:, 1]

#%% Plots
    
plt.plot(time, x_RK, label = "prey")
plt.plot(time, y_RK, label = "predator")
plt.legend(loc='middle left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Steady state populations')

#%% Phase plot

plt.plot(x_RK, y_RK, '-->', markevery=5, label = 'Phase Plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')


#%% Timestep 10 times shorter

T = 15.0            # Final time
dt = 0.001           # Time increment
N = int((T/dt)) + 1   # Number of time steps = Final time / time increment + 1 as an integer
x0 = 10             # Initial x value (prey?)
y0 = 2              # Initial y value (predator?)
t0 = 0.0              # Initial time (Not in any of the equations?)
alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
# The t0 is a seemingly useless variable because it should be defined as a relative start time


#%% Initial conditions
u_euler = np.empty((N,2))  # Create an empty array with N number of rows and 2 columns

# Array containing the solution
u_euler[0] = np.array([x0,y0])  # Defines the first row as an array = to x0 and y0

# Use for loop to solve for number of each species in each time step
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
    
time = np.linspace(t0, T+t0, N)  # Create array from initial time to end time with the number of time increments
x_euler = u_euler[:, 0]
y_euler = u_euler[:, 1]

plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Predator prey model')

#%% System behavior

plt.plot(x_euler, y_euler, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')
    
#%% RK method
alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
T = 15.0            
dt = 0.001
N = int(T/dt) + 1 
x0 = 10.
y0 = 2.
t0 = 0.     

u_RK = np.empty((N,2))
u_RK[0] = np.array([x0,y0])

for n in range(N-1):
    u_RK[n+1] = RK4(u_RK[n], f, dt)

time = np.linspace(t0, T+t0, N)  
x_RK = u_RK[:, 0]
y_RK = u_RK[:, 1]

#%% Plots
    
plt.plot(time, x_RK, label = "prey")
plt.plot(time, y_RK, label = "predator")
plt.legend(loc='middle left')
plt.xlabel('Time')
plt.ylabel('Number of each Species')
plt.title('Steady state populations')

#%% Phase plot

plt.plot(x_RK, y_RK, '-->', markevery=5, label = 'Phase Plot')
plt.legend(loc='upper right')
plt.xlabel('Number of prey')
plt.ylabel('Number of predators')
plt.title('Predator prey model')

"""
Making the timestep larger makes the euler equation significantly worse
Making the timestep smaller makes the approximation better
"""


#%% Lecture 10.2

# Import a bunch of packages for Python 3
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.mpl.ticker as cticker
import geopandas as gpd
from descartes import PolygonPatch

from mpl_toolkits.basemap import Basemap

#%% Exercise 1

"""
All of the packages loaded fine...
"""

#%% Loading shapefiles using geopandas

# Need a function to load a shapefile for some reason?

def load_shape_file(filepath):
    """
    Load the shapefile desired to mask a grid
    Arguments: 
        filepath: Path to *.shp file
    """
    
    shpfile = gpd.read_file(filepath)
    return shpfile

#%% Loading a shapefile
    
shp = load_shape_file("LMEs66.shp")
print(shp.head())
#shp 

#%% Printing the data table
print(shp)

#%% Exercise 2

print(shp.type)
print(shp.shape)

"""
The object itself is a dataframe of size (66,10)
Within the object, there are many polygons and multipolygons
"""

#%% Plotting a basic map using Baesmap

ax = plt.figure(figsize=(16,20), facecolor = 'w')

# Set the plot limits (the plot extent)
limN, limS, limE, limW = 84, -80, 180, -180

m = Basemap(projection='cyl', llcrnrlon=limW, \
            urcrnrlon = limE, llcrnrlat = limS, urcrnrlat=limN, resolution = 'c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#BDA973', lake_color='#BDA973')

#%% Plotting lat/long lines using basemap

ax = plt.figure(figsize=(16,20), facecolor = 'w')

# Set the plot limits (the plot extent)
limN, limS, limE, limW = 84, -80, 180, -180

m = Basemap(projection='cyl', llcrnrlon=limW, \
            urcrnrlon = limE, llcrnrlat = limS, urcrnrlat=limN, resolution = 'c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#BDA973', lake_color='#BDA973')

# Draw the latitude lines                 
parallels = np.arange(-90., 90., 20.)
m.drawparallels(parallels, labels=[1,0,0,0], fontsize = 10)

# Draw the longitude lines (meridians)
meridians = np.arange(-180., 180., 20.)
m.drawmeridians(meridians, labels = [0,0,0,1], fontsize = 10)

#%% Exercise 3a

ax = plt.figure(figsize=(16,20), facecolor = 'w')

# Set the plot limits (the plot extent)
limN, limS, limE, limW = 84, -80, 180, -180

m = Basemap(projection='cyl', llcrnrlon=limW, \
            urcrnrlon = limE, llcrnrlat = limS, urcrnrlat=limN, resolution = 'c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='green', lake_color='#BDA973')

# Draw the latitude lines                 
parallels = np.arange(-90., 90., 20.)
m.drawparallels(parallels, labels=[1,0,0,0], fontsize = 10)

# Draw the longitude lines (meridians)
meridians = np.arange(-180., 180., 20.)
m.drawmeridians(meridians, labels = [0,0,0,1], fontsize = 10)

#%% Exercise 3b

ax = plt.figure(figsize=(16,20), facecolor = 'w')

# Set the plot limits (the plot extent)
limN, limS, limE, limW = 84, -80, 180, -180

m = Basemap(projection='cyl', llcrnrlon=limW, \
            urcrnrlon = limE, llcrnrlat = limS, urcrnrlat=limN, resolution = 'c')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#BDA973', lake_color='blue')

# Draw the latitude lines                 
parallels = np.arange(-90., 90., 20.)
m.drawparallels(parallels, labels=[1,0,0,0], fontsize = 10)

# Draw the longitude lines (meridians)
meridians = np.arange(-180., 180., 20.)
m.drawmeridians(meridians, labels = [0,0,0,1], fontsize = 10)

#%% Exercise 3c

"""Resolutions
resolution of boundary database to use. 
Can be c (crude), l (low), i (intermediate), h (high), f (full) or None. 
If None, no boundary data will be read in (and class methods such as drawcoastlines will raise an if invoked). 
Resolution drops off by roughly 80% between datasets. 
Higher res datasets are much slower to draw. 
Default c. 
Coastline data is from the GSHHS (http://www.soest.hawaii.edu/wessel/gshhs/gshhs.html). 
State, country and river datasets from the Generic Mapping Tools (http://gmt.soest.hawaii.edu).
"""

ax = plt.figure(figsize=(16,20), facecolor = 'w')

# Set the plot limits (the plot extent)
limN, limS, limE, limW = 84, -80, 180, -180

m = Basemap(projection='cyl', llcrnrlon=limW, \
            urcrnrlon = limE, llcrnrlat = limS, urcrnrlat=limN, resolution = 'l')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='#BDA973', lake_color='#BDA973')

# Draw the latitude lines                 
parallels = np.arange(-90., 90., 20.)
m.drawparallels(parallels, labels=[1,0,0,0], fontsize = 10)

# Draw the longitude lines (meridians)
meridians = np.arange(-180., 180., 20.)
m.drawmeridians(meridians, labels = [0,0,0,1], fontsize = 10)

#%% Exercise 3d

ax = plt.figure(figsize=(16,20), facecolor = 'w')

# Set the plot limits (the plot extent)
limN, limS, limE, limW = 35, 15, -80, -100

m = Basemap(projection='cyl', llcrnrlon=limW, \
            urcrnrlon = limE, llcrnrlat = limS, urcrnrlat=limN, resolution = 'l')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='green', lake_color='blue')

# Draw the latitude lines                 
parallels = np.arange(15., 35., 5.)
m.drawparallels(parallels, labels=[1,0,0,0], fontsize = 10)

# Draw the longitude lines (meridians)
meridians = np.arange(-100., -80., 5.)
m.drawmeridians(meridians, labels = [0,0,0,1], fontsize = 10)

#%% Plotting shapefile?

# In this version of loading a shapefile, we do not need the file extension

sppath = 'LMEs66'

ax = plt.figure(figsize = (16,20), facecolor = 'w')

limN, limS, limE, limW = 84, -80, 180, -180

m = Basemap(projection='cyl', llcrnrlon=limW, \
            urcrnrlon = limE, llcrnrlat = limS, urcrnrlat=limN, resolution = 'l')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='green', lake_color='blue')

m.readshapefile(sppath, 'LME') # Second argument is the name of the shapefile data inside the shapefile

# Plot all the shapes in the shapefile
for info, shape in zip(m.LME_info, m.LME):
    x, y = zip(*shape)
    m.plot(x, y, marker = None, color = 'k', linewidth = '2')

#%% Exercise 4

ax = plt.figure(figsize = (16,20), facecolor = 'w')

limN, limS, limE, limW = 84, -80, 180, -180

m = Basemap(projection='cyl', llcrnrlon=limW, \
            urcrnrlon = limE, llcrnrlat = limS, urcrnrlat=limN, resolution = 'l')
m.drawcoastlines()
m.drawmapboundary()
m.fillcontinents(color='green', lake_color='blue')

m.readshapefile(sppath, 'LME') # Second argument is the name of the shapefile data inside the shapefile

# Plot all the shapes in the shapefile
for info, shape in zip(m.LME_info, m.LME):
    x, y = zip(*shape)
    m.plot(x, y, marker = None, color = 'red', linewidth = '1')

#%% Plotting shapefile polygons with cartopy and descartes

# Selects a part of the shapefile

def select_shape(shpfile, category, name):
    """Select the submask of interest from the shapefile.
    Args:
        geopandas shpfile object: (from *.shp) loaded through `load_shape_file`
        category: (str) header of shape file from which to filter shape.
            (Run print(shpfile) to see options)
        name: (str) name of shape relative to category.
        plot: (optional bool) if True, plot the polygon that will be masking.
    Returns:
        shapely polygon
    Example:
        from esmask.mask import load_shape_file, select_shape
        LME = load_shape_file('LMEs.shp')
        CalCS = select_shape(LME, 'LME_NAME', 'California Current')
    """
    s = shpfile
    polygon = s[s[category] == name]
    polygon = polygon.geometry[:].unary_union #magic black box off of stack exchange (should paste link), concatinating polygons
    return polygon

#%% Running the function. Exercise 5
    
CalCS_shp = select_shape(shp, 'LME_NAME', 'California Current')
print(CalCS_shp.type)
#print(CalCS_shp.shape)  Polygon files have no shape...
CalCS_shp

# The data type is a polygon

#%% Exercise 6

GOA_shp = select_shape(shp, 'LME_NAME', 'Gulf of Alaska')
GOA_shp

#%% Labrador Newfoundland

LN_shp = select_shape(shp, 'LME_NAME', 'Labrador - Newfoundland')
LN_shp

#%% Plot single polygon in shapefile as a patch on a map

def lat_lon_formatter(ax):
    """
    Creates nice latitude/longitude labels
    for maps
    """
    lon_formatter = cticker.LongitudeFormatter()
    lat_formatter = cticker.LatitudeFormatter()
    ax.xaxis.set_major_formatter(lon_formatter)
    ax.yaxis.set_major_formatter(lat_formatter)
    ax.tick_params(labelsize=16)
    
# the below function uses lat_lon_formatter above
def set_up_map(ax, x0, x1, y0, y1):
    """
    Adds coastline, etc.
    
    x0, x1: longitude bounds to zoom into
    y0, y1: latitude bounds to zoom into
    """
    # set up land overlay
    ax.add_feature(cfeature.LAND, facecolor='k')
    
    # zoom in on region of interest
    ax.set_extent([x0, x1, y0, y1])
    
    # set nicer looking ticks
    ax.set_xticks(np.arange(x0, x1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(y0, y1, 10), crs=ccrs.PlateCarree())
    lat_lon_formatter(ax)
    
#%% California Current demo
    
f, ax = plt.subplots(ncols=2, figsize = (10, 5), subplot_kw=dict(projection=ccrs.PlateCarree()))
# Set the projection within the plot

set_up_map(ax[0], -140, -107, 20, 50)
set_up_map(ax[1], -140, -107, 20, 50)

#%% Exercise 7

def set_up_map(ax, x0, x1, y0, y1):
    """
    Adds coastline, etc.
    
    x0, x1: longitude bounds to zoom into
    y0, y1: latitude bounds to zoom into
    """
    # set up land overlay
    ax.add_feature(cfeature.LAND, facecolor='green')
    
    # zoom in on region of interest
    ax.set_extent([x0, x1, y0, y1])
    
    # set nicer looking ticks
    ax.set_xticks(np.arange(x0, x1, 10), crs=ccrs.PlateCarree())
    ax.set_yticks(np.arange(y0, y1, 10), crs=ccrs.PlateCarree())
    lat_lon_formatter(ax)

f, ax = plt.subplots(ncols=2, figsize = (10, 5), subplot_kw=dict(projection=ccrs.PlateCarree()))

set_up_map(ax[0], -140, -107, 20, 50)
set_up_map(ax[1], -140, -107, 20, 50)

#%% Plot CA current LME polygon

from descartes import PolygonPatch

help(PolygonPatch)

#%% 

f, ax = plt.subplots(ncols=2, figsize = (10, 5), subplot_kw=dict(projection=ccrs.PlateCarree()))

set_up_map(ax[0], -140, -107, 20, 50)
set_up_map(ax[1], -140, -107, 20, 50)

ax[0].add_patch(PolygonPatch(CalCS_shp, fc='#add8e6'))  # PolygonPatch can also take other matplotlib commands
ax[1].add_patch(PolygonPatch(CalCS_shp, fc='none', ec = 'r', linewidth=2, linestyle = ':'))  # Use None for the facecolor to specify no color

#%% Plot the things on top of each other

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1, projection = ccrs.PlateCarree())
set_up_map(ax, -140, -107, 20, 50)

# Plot shapefile in blue with red dotted boundary
ax.add_patch(PolygonPatch(CalCS_shp, fc = 'None', ec = 'r', linewidth = 2, linestyle = ':'))
ax.add_patch(PolygonPatch(CalCS_shp, fc = '#add8e6', ec = 'None', alpha = 1, zorder = 0))

"""
ec = edge color
fc = fill color
"""

#%% Again

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1, projection = ccrs.PlateCarree())
set_up_map(ax, -140, -107, 20, 50)

# Plot shapefile in blue with red dotted boundary
ax.add_patch(PolygonPatch(CalCS_shp, fc = '#add8e6', ec = 'r', linewidth = 2, linestyle = ':'))
                          
# No need to have 2 polygons to layer the same polygon on top of itself

#%% Exercise 8
fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1, projection = ccrs.PlateCarree())
set_up_map(ax, -140, -107, 20, 50)

# Plot shapefile in blue with red dotted boundary
ax.add_patch(PolygonPatch(CalCS_shp, fc = 'None', ec = 'r', linewidth = 2, linestyle = ':', zorder = 1))
ax.add_patch(PolygonPatch(CalCS_shp, fc = '#add8e6', alpha = 1, zorder = 10))

"""
Removing the ec argument goes back to the default edge color
Zorder changes the order of each patch within the plot as a whole. Higher values are placed on top of lower values
"""

#%% Exercise 9

"""
Required parts
Gulf of Alaska
California Current
Gulf of California
Pacific Central-American Coastal
Humboldt Current
"""

CalCS_shp = select_shape(shp, 'LME_NAME', 'California Current')
GOA_shp = select_shape(shp, 'LME_NAME', 'Gulf of Alaska')
GOC_shp = select_shape(shp, 'LME_NAME', 'Gulf of California')
PCAC_shp = select_shape(shp, 'LME_NAME', 'Pacific Central-American Coastal')
HC_shp = select_shape(shp, 'LME_NAME', 'Humboldt Current')

fig = plt.figure(figsize = (5,5))
ax = fig.add_subplot(1,1,1, projection = ccrs.PlateCarree())
set_up_map(ax, -180, -45, 85, -85)

ax.add_patch(PolygonPatch(GOA_shp, fc= 'orange', alpha = 1, zorder = 0))
ax.add_patch(PolygonPatch(CalCS_shp, fc = 'red', alpha = 1, zorder = 1))
ax.add_patch(PolygonPatch(GOC_shp, fc = 'blue', alpha = 1, zorder = 2))
ax.add_patch(PolygonPatch(PCAC_shp, fc = 'gray', alpha = 1, zorder = 3))
ax.add_patch(PolygonPatch(HC_shp, fc = 'purple', alpha = 1, zorder = 4))
