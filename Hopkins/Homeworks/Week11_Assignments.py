# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 14:00:28 2019

@author: Miles
"""
#%%
"""
Warp Up 10.1
"""
#%%
import numpy
import matplotlib.pyplot as plt

#%%
"""
Exercise 1:
    Plot the fixed point (steady populations) of the system on both the
    timeseries and phase space plots. Describe where the fixed point is in the 
    phase space plot.
"""
#%%
y_steady = alpha / beta
x_steady = gamma / beta
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_euler, label = 'prey ')
plt.plot(time, y_euler, label = 'predator')
plt.plot([0, 15], [x_steady, x_steady])
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("predator prey model")
#%%
plt.plot(x_euler, y_euler, '-->', markevery=5, label = 'phase plot')
plt.plot(x_steady, y_steady, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')

#labels
plt.xlabel("number of prey")
plt.ylabel("number of predators")

#title
plt.title("predator prey model")
#%%
"""
Exercise 1.5:
    If you start the system with the steady state, what happens?
    
Answer:
    Instead of spiraling larger and larger, both populations fluctuate between
    minimum and maximum populations, never growing larger or smaller. This
    allows neither species to truly reach 0 and keeps them from gaining
    extremes.
"""
#%%
# set time-increment and discretize the time
T  = 15.0                           # final time
dt = 0.01                           # set time-increment
N  = int(T/dt) + 1                  # number of time-steps
x0 = x_steady
y0 = y_steady
t0 = 0.

# set initial conditions
u_euler = numpy.empty((N, 2))

# initialize the array containing the solution for each time-step
u_euler[0] = numpy.array([x0, y0])

# use a for loop to call the function rk2_step()
for n in range(N-1):
    
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = numpy.linspace(t0, t0+T,N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_euler, label = 'prey')
plt.plot(time, y_euler, label = 'predator')
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("steady state values")
#%%
"""
Exercise 2:
    Vary the interactions between species by changing alpha, beta, gamma, and
    [the fuck is that squiggle??] and see what happens to the pop. evolution
    as well as phase plots.
"""
#%%
# set the initial parameters
# alpha = 1.
# beta = 1.2
# gamma = 4.
# delta = 1.
#%% experiment parameters
# change alpha
alpha = 4. # Pop. growth of prey
beta = 1.2 # Rate at which predator and prey meet
gamma = 4. # Death rate of predators
delta = 1. # Rate of change
#%% experiment parameters
# change beta
alpha = 1. # Pop. growth of prey
beta = 4. # Rate at which predator and prey meet
gamma = 4. # Death rate of predators
delta = 1. # Rate of change
#%% experiment parameters
# change gamma
alpha = 1. # Pop. growth of prey
beta = 1.2 # Rate at which predator and prey meet
gamma = 2. # Death rate of predators
delta = 1. # Rate of change
#%% experiment parameters
# change delta
alpha = 1. # Pop. growth of prey
beta = 1.2 # Rate at which predator and prey meet
gamma = 4. # Death rate of predators
delta = 3. # Rate of change
#%%
#define the time stepping scheme - euler forward, as used in earlier lessons
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
#%%
# define the function that represents the Lotka-Volterra equations
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
    return numpy.array([x*(alpha - beta*y), -y*(gamma - delta*x)])
#%%
# set time-increment and discretize the time
T  = 15.0                           # final time
dt = 0.01                           # set time-increment
N  = int(T/dt) + 1                  # number of time-steps
x0 = 10.
y0 = 2.
t0 = 0.

# set initial conditions
u_euler = numpy.empty((N, 2))

# initialize the array containing the solution for each time-step
u_euler[0] = numpy.array([x0, y0])

# use a for loop to call the function rk2_step()
for n in range(N-1):
    
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = numpy.linspace(t0, t0+T, N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_euler, label = 'prey ')
plt.plot(time, y_euler, label = 'predator')
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("predator prey model")
#%%
"""
Exercise 3:
    Do the same exercise with a fourth order time stepping method called
    "Runge-Kutta 4" whose algorithm is given below. Do your answers differ?
"""
#%%
def RK4(u,f,dt):
    # Runge Kutta 4th order method
    """Returns the solution at the next time-step using Runge Kutta fourth
    order (RK4) method.
    
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
#%%
# define the function that represents the Lotka-Volterra equations
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
    return numpy.array([x*(alpha - beta*y), -y*(gamma - delta*x)])
#%%
# set time-increment and discretize the time
T  = 15.0                           # final time
dt = 0.01                           # set time-increment
N  = int(T/dt) + 1                  # number of time-steps
x0 = 10
y0 = 2
t0 = 0.

# set initial conditions
u_rk4 = numpy.empty((N, 2))

# initialize the array containing the solution for each time-step
u_rk4[0] = numpy.array([x0, y0])

# use a for loop to call the function rk2_step()
for n in range(N-1):
    
    u_rk4[n+1] = RK4(u_rk4[n], f, dt)
#%%
time = numpy.linspace(t0, t0+T, N)
x_rk4 = u_rk4[:,0]
y_rk4 = u_rk4[:,1]
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_rk4, label = 'prey ')
plt.plot(time, y_rk4, label = 'predator')
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("predator prey model")
#%%
plt.plot(x_rk4, y_rk4, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
#labels
plt.xlabel("number of prey")
plt.ylabel("number of predators")
#title
plt.title("predator prey model")
#%%
y_steady = alpha / beta
x_steady = gamma / beta
#%%
plt.plot(x_rk4, y_rk4, '-->', markevery=5, label = 'phase plot')
plt.plot(x_steady, y_steady, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
#labels
plt.xlabel("number of prey")
plt.ylabel("number of predators")
#title
plt.title("predator prey model")
#%%
"""
Warm Up 10.2
"""
#%%
""" Numerical Solutions Using Python """
"""
A simple python code for solving these equations is shown below.
"""
#%%
# set the initial parameters
# alpha = 1.
# beta = 1.2
# gamma = 4.
# delta = 1.
alpha = 1. # Pop. growth of prey
beta = 1.2 # Rate at which predator and prey meet
gamma = 4. # Death rate of predators
delta = 1. # Rate of change
#%%
#define the time stepping scheme - euler forward, as used in earlier lessons
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
#%%
# define the function that represents the Lotka-Volterra equations
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
    return numpy.array([x*(alpha - beta*y), -y*(gamma - delta*x)])
#%%
"""
Exercise 1:
    Make the timestamp 10x bigger.
Answer:
    It makes the Euler method far worse. It is far less stable and provides 
    poor insights or answers. It is effectively now useless.
"""
#%% 10x more
#%%
# set time-increment and discretize the time
T  = 15.0                           # final time
dt = 0.1                          # set time-increment
N  = int(T/dt) + 1                  # number of time-steps
x0 = 10.
y0 = 2.
t0 = 0.

# set initial conditions
u_euler = numpy.empty((N, 2))

# initialize the array containing the solution for each time-step
u_euler[0] = numpy.array([x0, y0])

# use a for loop to call the function rk2_step()
for n in range(N-1):
    
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = numpy.linspace(0.0, T,N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_euler, label = 'prey ')
plt.plot(time, y_euler, label = 'predator')
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("predator prey model")
#%%
plt.plot(x_euler, y_euler, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
#labels
plt.xlabel("number of prey")
plt.ylabel("number of predators")
#title
plt.title("predator prey model")
#%%
"""
Exercise 2:
    Make it 10x less.
Answer:
    This makes the Euler method far more accurate, it no longer has its death
    spiral within the phase plot.
"""
#%% 10x less
#%%
# set time-increment and discretize the time
T  = 15.0                           # final time
dt = 0.001                          # set time-increment
N  = int(T/dt) + 1                  # number of time-steps
x0 = 10.
y0 = 2.
t0 = 0.

# set initial conditions
u_euler = numpy.empty((N, 2))

# initialize the array containing the solution for each time-step
u_euler[0] = numpy.array([x0, y0])

# use a for loop to call the function rk2_step()
for n in range(N-1):
    
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = numpy.linspace(0.0, T,N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_euler, label = 'prey ')
plt.plot(time, y_euler, label = 'predator')
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("predator prey model")
#%%
plt.plot(x_euler, y_euler, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
#labels
plt.xlabel("number of prey")
plt.ylabel("number of predators")
#title
plt.title("predator prey model")
#%%
"""
Exercise 3:
    Try the same thing with the RK4 method.
Answer:
    Even with these changes the RK4 method barely changes, in either case. It
    really just gains or loses points and that is effectively it. It seems
    like a much more stable system the Euler.
"""
#%%
def RK4(u,f,dt):
    # Runge Kutta 4th order method
    """Returns the solution at the next time-step using Runge Kutta fourth
    order (RK4) method.
    
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
#%%
# define the function that represents the Lotka-Volterra equations
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
    return numpy.array([x*(alpha - beta*y), -y*(gamma - delta*x)])
#%% 10x more RK4
#%%
# set time-increment and discretize the time
T  = 15.0                           # final time
dt = 0.1                           # set time-increment
N  = int(T/dt) + 1                  # number of time-steps
x0 = 10
y0 = 2
t0 = 0.

# set initial conditions
u_rk4 = numpy.empty((N, 2))

# initialize the array containing the solution for each time-step
u_rk4[0] = numpy.array([x0, y0])

# use a for loop to call the function rk2_step()
for n in range(N-1):
    
    u_rk4[n+1] = RK4(u_rk4[n], f, dt)
#%%
time = numpy.linspace(t0, t0+T, N)
x_rk4 = u_rk4[:,0]
y_rk4 = u_rk4[:,1]
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_rk4, label = 'prey ')
plt.plot(time, y_rk4, label = 'predator')
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("predator prey model")
#%%
plt.plot(x_rk4, y_rk4, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
#labels
plt.xlabel("number of prey")
plt.ylabel("number of predators")
#title
plt.title("predator prey model")
#%%
#%% 10x less RK4
#%%
# set time-increment and discretize the time
T  = 15.0                           # final time
dt = 0.001                           # set time-increment
N  = int(T/dt) + 1                  # number of time-steps
x0 = 10
y0 = 2
t0 = 0.

# set initial conditions
u_rk4 = numpy.empty((N, 2))

# initialize the array containing the solution for each time-step
u_rk4[0] = numpy.array([x0, y0])

# use a for loop to call the function rk2_step()
for n in range(N-1):
    
    u_rk4[n+1] = RK4(u_rk4[n], f, dt)
#%%
time = numpy.linspace(t0, t0+T, N)
x_rk4 = u_rk4[:,0]
y_rk4 = u_rk4[:,1]
#%%
# We will now plot the variation of population for each species with time.
plt.plot(time, x_rk4, label = 'prey ')
plt.plot(time, y_rk4, label = 'predator')
plt.legend(loc='upper right')

#labels
plt.xlabel("time")
plt.ylabel("number of each species")

#title
plt.title("predator prey model")
#%%
plt.plot(x_rk4, y_rk4, '-->', markevery=5, label = 'phase plot')
plt.legend(loc='upper right')
#labels
plt.xlabel("number of prey")
plt.ylabel("number of predators")
#title
plt.title("predator prey model")
#%%
"""
10.2 Exercises
"""
#%%
"""
Exercise 2:
    What type of object is shp?
Answer:
    geopandas.geodataframe.GeoDataFrame
"""
#%%
type(shp)
#%%
