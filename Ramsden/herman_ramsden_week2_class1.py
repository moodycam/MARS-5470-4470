# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:17:31 2019

@author: srv_veralab
"""

#%% 

def func0():
    print("test")

func0()

#%%

def func1(s):
    """Print a string 's' and tell how many characters it has
    """
    print(s + " has " + str(len(s)) + " characters ")
func1("test")

#%%
x = 1
word = ['word']
#%%

def square(x):
    return x ** 2
square(3)

#%%

def powers(x):
    return x**2, x**3, x**4
powers(3)

#%%

def myfunc(x, p=2, debug=True):
    if debug:
        print("evaluating myfunc for x = " + str(x) + "using exponent p = " + str(p))
        return x**p
    
myfunc(7)

#%%

#Unnamed functions (lambda function)

f1 = lambda x : x**2

# is equivalent to 

def f2(x):
    return x**2

f1(2), f2(2)

#%%

x = [2,4,6,8,12]

def average(x):
    return ((sum(x))/len(x))

average(x)

#%%

y = [2,4,6]

def highest(y):
    return max(y)

highest(y)

#%%


