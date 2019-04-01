#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 13:37:55 2019

@author: roryeggleston
"""

#population growth: rabbit example + exponential growth
# pop = N, time = t, N(t) = No e^at
#dN/dt = delta N/delta time -->rate of population growth: the more rabbits there are, the more rabbits there are
#Lotka-Volterra equations:
#prey: dx/dt = x(a-By) --> dx/dt = ax - Bxy (last term: predation term, encounter rate of x and y, some portion B is eaten)
#predator: dy/dt = -y(g-dx) --> dy/dt = dxy - gy (last term is predator mortality)
import numpy as np
import matplotlib.pyplot as plt
#%%
alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
#%%
def euler_step(u, f, dt):
    return u + dt * f(u)
#%%
def f(u):
    x = u[0]
    y = u[1]
    return np.array([x*(alpha -beta*y), -y*(gamma - delta*x)])
#%%
T = 15.0
dt = 0.01
N = int(T/dt) + 1
x0 = 10.
y0 = 2.0
t0 = 0.

u_euler  = np.empty((N,2))
u_euler[0] = np.array([x0, y0])
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = np.linspace(0.0, T, N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
plt.plot(x_euler, y_euler, '-->', markevery = 5, label = "Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of prey")
plt.ylabel("# of predators")
plt.title("Predator-Prey Model")
#%%
#EXERCISE 1
#Steady State
YSS = alpha/beta
XSS = gamma/beta
steady_statex = np.linspace(XSS, XSS, N)
steady_statey = np.linspace(YSS, YSS, N)
#%%
#Timeseries
plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.plot(time, steady_statex, label = 'Prey Steady State')
plt.plot(time, steady_statey, label = 'Predator Steady State')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
#Phase Space
plt.plot(x_euler, y_euler, '-->', markevery = 5, label = "Phase Plot")
plt.plot(steady_statex, steady_statey, "-->", markevery = 5, label = "Steady State Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of prey")
plt.ylabel("# of predators")
plt.title("Predator-Prey Model")
#The fixed point in the phase space plot is a single point in the empty space between the predator-prey phase space plot, indidcating that if the prey pop stays the same, so does the predator pop.
#%%
#1.5
T = 15.0
dt = 0.01
N = int(T/dt) + 1
x0 = XSS
y0 = YSS
t0 = 0.

u_euler  = np.empty((N,2))
u_euler[0] = np.array([x0, y0])
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = np.linspace(0.0, T, N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
#EXERCISE 2
alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
#%%
def euler_step(u, f, dt):
    return u + dt * f(u)
#%%
def f(u):
    x = u[0]
    y = u[1]
    return np.array([x*(alpha -beta*y), -y*(gamma - delta*x)])
#%%
T = 15.0
dt = 0.01
N = int(T/dt) + 1
x0 = 10.
y0 = 2.0
t0 = 0.

u_euler  = np.empty((N,2))
u_euler[0] = np.array([x0, y0])
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = np.linspace(0.0, T, N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
plt.plot(x_euler, y_euler, '-->', markevery = 5, label = "Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of prey")
plt.ylabel("# of predators")
plt.title("Predator-Prey Model")
#%%
#EXERCISE 3
def RK4(u,f,dt):
    k1 = f(u)
    u1 = u + (dt/2.)*k1
    k2 = f(u1)
    u2 = u + (dt/2.)*k2
    k3 = f(u2)
    u3 = u + dt*k3
    k4 = f(u3)
    return u + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
#%%
def f(u):
    x = u[0]
    y = u[1]
    return np.array([x*(alpha -beta*y), -y*(gamma - delta*x)])
#%%
T = 15.0
dt = 0.01
N = int(T/dt) + 1
x0 = 10.
y0 = 2.0
t0 = 0.

u_runge  = np.empty((N,2))
u_runge[0] = np.array([x0, y0])
for n in range(N-1):
    u_runge[n+1] = RK4(u_runge[n], f, dt)
#%%
time = np.linspace(0.0, T, N)
x_runge = u_runge[:,0]
y_runge = u_runge[:,1]
#%%
plt.plot(time, x_runge, label = 'Prey')
plt.plot(time, y_runge, label = 'Predator')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
plt.plot(x_runge, y_runge, '-->', markevery = 5, label = "Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of prey")
plt.ylabel("# of predators")
plt.title("Predator-Prey Model")
#%%
#EXERCISE 4
from scipy.integrate import odeint
plt.ion()
plt.rcParams['figure.figsize'] = 10, 8 

P = 0
d = 0.0001
B = 0.0095
G = 0.0001
A = 0.0001

def f(y, t):
    Si = y[0]
    Zi = y[1]
    Ri = y[2]
    
    f0 = P - B*Si*Zi -d*Si
    f1 = B*Si*Zi + G*Ri - A*Si*Zi
    f2 = d*Si + A*Si - G*Ri
    return [f0, f1, f2]
S0 = 500.
Z0 = 0
R0= 0
y0 = [S0, Z0, R0]
t = np.linspace(0, 5., 1000)

soln = odeint(f, y0, t)
S = soln[:, 0]
Z = soln[:, 1]
R = soln[:, 2]

plt.figure()
plt.plot(t, S, label='Living')
plt.plot(t, Z, label='Zombies')
plt.xlabel('Days from outbreak')
plt.ylabel('Population')
plt.title('Zombie Apocalypse')
plt.legend(loc=0)
#%%
#Playing around
plt.ion()
plt.rcParams['figure.figsize'] = 10, 8 

P = 0.001
d = 0.001
B = 0.001
G = 0.001
A = 0.0001

def f(y, t):
    Si = y[0]
    Zi = y[1]
    Ri = y[2]
    
    f0 = P*Si - B*Si*Zi -d*Si
    f1 = B*Si*Zi + G*Ri - A*Si*Zi
    f2 = d*Si + A*Si - G*Ri
    return [f0, f1, f2]
S0 = 500000.
Z0 = 1
R0= 0
y0 = [S0, Z0, R0]
t = np.linspace(0, 50., 1000)

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
plt.title('Zombie Apocalypse')
plt.legend(loc=0)
#%%
#ZOMBIE PHASE SPACE
plt.plot(S, R, '-->', markevery = 5, label = "Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of humans")
plt.ylabel("# of zombies")
plt.title("Zombie Model")
#%%
#WARM UP 032719
import numpy as np
import matplotlib.pyplot as plt
#%%
alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
#%%
def euler_step(u, f, dt):
    return u + dt * f(u)
#%%
def f(u):
    x = u[0]
    y = u[1]
    return np.array([x*(alpha -beta*y), -y*(gamma - delta*x)])
#%%
T = 15.0
dt = 0.1
N = int(T/dt) + 1
x0 = 10.
y0 = 2.0
t0 = 0.

u_euler  = np.empty((N,2))
u_euler[0] = np.array([x0, y0])
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = np.linspace(0.0, T, N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
plt.plot(x_euler, y_euler, '-->', markevery = 5, label = "Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of prey")
plt.ylabel("# of predators")
plt.title("Predator-Prey Model")
#The solution definitely changes, and makes the figures even more nonsensical (ie introcucing negatives, etc)
#The differences are noticeable in both, but much more obvious in the phase space plot.
#%%
def RK4(u,f,dt):
    k1 = f(u)
    u1 = u + (dt/2.)*k1
    k2 = f(u1)
    u2 = u + (dt/2.)*k2
    k3 = f(u2)
    u3 = u + dt*k3
    k4 = f(u3)
    return u + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
#%%
def f(u):
    x = u[0]
    y = u[1]
    return np.array([x*(alpha -beta*y), -y*(gamma - delta*x)])
#%%
T = 15.0
dt = 0.1
N = int(T/dt) + 1
x0 = 10.
y0 = 2.0
t0 = 0.

u_runge  = np.empty((N,2))
u_runge[0] = np.array([x0, y0])
for n in range(N-1):
    u_runge[n+1] = RK4(u_runge[n], f, dt)
#%%
time = np.linspace(0.0, T, N)
x_runge = u_runge[:,0]
y_runge = u_runge[:,1]
#%%
plt.plot(time, x_runge, label = 'Prey')
plt.plot(time, y_runge, label = 'Predator')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
plt.plot(x_runge, y_runge, '-->', markevery = 5, label = "Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of prey")
plt.ylabel("# of predators")
plt.title("Predator-Prey Model")
#The RK4 method solution does not change much if you increase the timestep by 10 (except for taking a longer time between cycles). This is true of both the timeseries and the phase space.
#%%
alpha = 1.
beta = 1.2
gamma = 4.
delta = 1.
#%%
def euler_step(u, f, dt):
    return u + dt * f(u)
#%%
def f(u):
    x = u[0]
    y = u[1]
    return np.array([x*(alpha -beta*y), -y*(gamma - delta*x)])
#%%
T = 15.0
dt = 0.001
N = int(T/dt) + 1
x0 = 10.
y0 = 2.0
t0 = 0.

u_euler  = np.empty((N,2))
u_euler[0] = np.array([x0, y0])
for n in range(N-1):
    u_euler[n+1] = euler_step(u_euler[n], f, dt)
#%%
time = np.linspace(0.0, T, N)
x_euler = u_euler[:,0]
y_euler = u_euler[:,1]
#%%
plt.plot(time, x_euler, label = 'Prey')
plt.plot(time, y_euler, label = 'Predator')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
plt.plot(x_euler, y_euler, '-->', markevery = 5, label = "Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of prey")
plt.ylabel("# of predators")
plt.title("Predator-Prey Model")
#Decreasing the timestep by a factor of 10 increases the quality of the solution for the euler method.
#%%
def RK4(u,f,dt):
    k1 = f(u)
    u1 = u + (dt/2.)*k1
    k2 = f(u1)
    u2 = u + (dt/2.)*k2
    k3 = f(u2)
    u3 = u + dt*k3
    k4 = f(u3)
    return u + (dt/6.)*(k1 + 2.*k2 + 2.*k3 + k4)
#%%
def f(u):
    x = u[0]
    y = u[1]
    return np.array([x*(alpha -beta*y), -y*(gamma - delta*x)])
#%%
T = 15.0
dt = 0.001
N = int(T/dt) + 1
x0 = 10.
y0 = 2.0
t0 = 0.

u_runge  = np.empty((N,2))
u_runge[0] = np.array([x0, y0])
for n in range(N-1):
    u_runge[n+1] = RK4(u_runge[n], f, dt)
#%%
time = np.linspace(0.0, T, N)
x_runge = u_runge[:,0]
y_runge = u_runge[:,1]
#%%
plt.plot(time, x_runge, label = 'Prey')
plt.plot(time, y_runge, label = 'Predator')
plt.legend(loc='upper right')
plt.xlabel("Time")
plt.ylabel("# of each species")
plt.title("Predator-Prey Model")
#%%
plt.plot(x_runge, y_runge, '-->', markevery = 5, label = "Phase Plot")
plt.legend(loc='upper right')
plt.xlabel("# of prey")
plt.ylabel("# of predators")
plt.title("Predator-Prey Model")
#Decreasing the timestep also minorly changed the RK4 solution, but not as dramatically as the euler solution.