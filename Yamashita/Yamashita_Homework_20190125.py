# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 15:47:50 2019

@author: tjyamashta_dell
"""

# Scipy functions

def test():
    print('in test function')
    
test()

def disk_area(radius):
    return 3.14 * radius * radius

disk_area(1.5)

def double_it(x):
    return x * 2

double_it(3)

double_it()

def double_it(x=2):
    return x * 2

double_it()
double_it(3)

def slicer(seq, start=None, stop=None, step=None):
    """Implement basic python slicing."""
    return seq[start:stop:step]

rhyme = 'one fish, two fish, red fish, blue fish'.split()
rhyme

slicer(rhyme)
slicer(rhyme, step=2)
slicer(rhyme, 1, step=2)
slicer(rhyme, start=1, stop=4, step=2)
slicer(rhyme, step=2, start=1, stop=4)

def try_to_modify(x, y, z):
    x = 23
    y.append(42)
    z = [99]
    print(x)
    print(y)
    print(z)
    
a = 77
b = [99]
c = [28]
try_to_modify(a,b,c)
print(a)
print(b)
print(c)

# Global variables

x = 5
def addx(y):
    return x + y

addx(10)

def setx(y):
    x = y
    print('x is %d' % x)
    
setx(10)
x

def setx(y):
    global x
    x = y
    print('x is %d' % x)

setx(10)
x

def variable_args(*args, **kwargs):
    print ('args is', args)
    print ('kwargs is', kwargs)
    
variable_args('one', 'two', x=1, y=2, z=3)

# Not sure what this does as the output seems to be different than what tutorial states
def funcname(params):
    """Concise one-line sentence describing the function. 
    
    Extended summary which can contain multiple paragraphs.
    """
    # function body
    pass

funcname?

# Can rename functions and store them as objects
va = variable_args
va('three', x=1, y=2)

#%% Excercises

# Source: 
    # https://www.w3resource.com/python-exercises/python-functions-exercises.php

## Question 3
def multiply_func(list):
    total = 1
    for value in list:
        total *= value
    return total

# Notes: These symbols can be used as mathamatical operators in functions: 
    # += (add over a series of values) , *= (multiply over a series of values)

#%% Testing functions
multiply_func([1,2,-3,4])

#%% Exercises

# Same source
# Question 5


def factorial(a):
    x = 1
    for i in range(1,a+1):
        x = x * i
    return x
factorial(4)

#%% Exercises

# Same source
# Question 6

# This works but is kind of useless because it just checks if one number is between 2 other numbers
def check_range(x, a, b):
   """
   x is value to chaeck in range
   a is lower end of range
   b is upper end of range
   """
    if (x >= a):
        if (x <= b):
            return "In Range"
        else:
            return "Greater than range"
    else:
        return "Less than range"
        

x = 0
y = range(1,50)
check_range(x,1,50)
