# -*- coding: utf-8 -*-
"""
Created on Sat Jan 19 15:22:20 2019

@author: tomyamashita
"""
# All practice code for MARS5470 - Introduction to Scientific Programming
# To Create a new cell: use 
    # '#%%' 
# without the quotations
    
###  TO GET HELP ON AN OBJECT, Ctrl+i with cursor on object

#%% 2019-01-14 Code and Notes

## To Return/execute code in JupyterLab, type shift->enter
## To Return/execute code in Spyder, us F9

a = 3
b = 4
print(a)  # When working in cells, to output values, you must use the print function
print(b)
a + b 
# End Cell

# a and b are integers in this case because no decimals


blah = "yes"
blah + blah + blah

# Defines an object, "blah" then concatenates values together
# Can add a space by using ' ' as part of concatenate function
# single quotes and double quotes are equivalent

# Lists
test = [1,2,3,4,5]
print(type(test))  # Use the type function to see what kind of value your object is
# This is a list

# Lists start at value 0 and go up not 1 as would be expected
# Use negative numbers to count backwards where -1 is the last value

# If you change the value of an object, be sure to rerun the object before running code dependent on it
colors = ['red','blue','green','black','white']
print(colors[0:2])
print(colors[-1])
print(colors[:2])
print(colors[0,3])
colors

# Note that item with index 2 does not show up so only item 0 and item 1 
# Syntax notes: first in syntax (before colon) is the starting item and second number (after colon) is the number of items to include 
# Syntax notes: object[start:stop:stride] where start is first value, stop is number of items to include, stride is by what number to include so object[0:5:1] is from first item to item 5 by 1

print(colors) #Print all the colors
print(colors[3:]) #Print from item 3 - end
print(colors[:3]) #Print from beginning and 3 total items
colors[0:5:2]  #Print from beginning to end by 2

colors2 = colors
colors2[0] = 'yellow'  # Replaces item 0 with new item
colors2


#%% 2019-01-16 Code and Notes

# Load the package called "math"
## Use the import function to import packages
import math

# Can load things within packages by using "package name"."function within package"

x = math.cos(2*math.pi)
print(x)

# Import all the functions from the package using the *
# Syntax is import from a package this thing
# * is everything
from math import *

x = cos(2*pi)
print(x)

cos(1)
math.cos(1)

# Need to use the package prefix because multiple packages may have the same function name so it is a way to distinguish which function

# Import a package and give it a new name
import numpy as np 

2>3  # Logical statement
3>2  # Logical statement

# Using the if logical statement. Similar to excel if statement
# Can use elif as an intermediary between if and else
# Program will choose the first true statement

if (2**2 == 5):
    print('Obvious')
elif (2**2 == 6):
    print('heck no')
else:
    print('no...')

# Note, tab thing affects everything in the script
if (2**2 == 5):
    print('obvious')
elif (2**2 != 6):
    print('heck no')
else:
    print('no')
    
# Logicals: != (not equal to), == (equal), <= (less than or equal to), << (Less than), >> (Greater than)
    
for i in range(4):
    print(i)

# i index is just a reference placeholder
# range function creates a range of values from 0 to value inputted

for word in ('cool', 'powerful', 'readable'):
    print('Python is %s' % word)
    
for value in range(4):
    print(cos(range(4)[value]))
    
a = range(16)
for i in a:
    print(a[i]**2 + a[i])
    
a[1]

mammals = ('bobcat', 'ocelot', 'raccoon', 'opossum', 'deer', 'nilgai')
for i in mammals:
    if (i == ('ocelot'):
        print('yes')
    else:
        print('no')

# Set value within the object you are iterating. Example: i is a value in mammals so you want to see if a value is equal to x not the object

for i in [1,3,5]:
    print(mammals[i])
    
for i in mammals[1::2]:
    print(i)

b = range(100)
for i in b[::10]:
    print(b[i])
    
mammals2 = ['bobcat', 'ocelot', 'raccoon', 'opossum', 'deer', 'nilgai']
for i in mammals2[::2]:
    print(i)
    
# What is the difference between [] and () when defining an object?
    # Look up online to see if can find
    # See these websites: 
        # https://www.quora.com/What-is-the-difference-between-and-in-Python-4
        # https://www.afternerd.com/blog/difference-between-list-tuple/ 
    
z = 1 +1j
while abs(z) < 100: 
    z = z**2 + 1
z

# Defines an equation
# Solves the equation and resolves the equation until the value reaches a threshold

a = 2
while (a < 5):
    print(a)
    a = a+1
    print(a)
    
# Cannot print numerical and strings
    
a = -1
while (a <5):
    if a == 0:
        break
    print(a)
    a = a + 1
    print(a)
    
# The break stops the loop at a particular value
    
#%% 2019-01-23

# Code playing before class
    
b = 0
while (b < 10): 
    if b == 5:
        break
    b = b**2 + 1
    print(b)
    
# Question: Concatenating strings + numbers?
    
c = [-3,-2,-1,0,1,2,3]
for value in c:
    if value == 0:
        continue
    print(3/value)

d = range(500)
type(d)
for value in d:
    if value == 100:
        break
    print(d*10)
    
# Above doesn't work
# Question: List of sequence instead of range as range seems to have certain limitations to how it can be used
x = [0:5]  # Doesn't work

#%% 2019-01-23

# Class Notes
## Variable assignments
(variable) = (assignment operator)
# Assignment operator can be of different types: 
    #integer (number without decimals)
    #float (number with decimals) 
    #string (use quotes to create strings)
    #boolean (logical statements)
    #lists: ['red','yellow','blue']: mutable list (can change)
    #tuple: ('red','yellow','blue'): immutable list (cannot change)
## Indexing
    (variable)[index value]
    # index value is a value number in a list, starting at 0 - infinity
    (variable)[:] #print whole list
    (variable)[:1] #print only first item in list = from first value 1 value
    (variable)[x:y:z] #x = first value to get, y = number of values past first value to look at, z = interval to grab values
    (variable)[start:stop:stride]
    
x = [1,2,3,4,5,6,7,8,9,10]
x[0:5:2]

## Syntax = punctuation and spelling of a coding language 

## Logical Control
### For loops
# when you want to repeat a function for a bunch of values
# Good for running functions with lists

### While Loops
# while (this is true)
    # do this
    # (Should include some sort of function that will change the value of the variable)

### If statements
#if (this statement is true):
    #do this
    #elif (if value of if statement is false) (This is an intermediary between if and else)
        #do this
    #else (if if and elif are all false)
        # do this

## Importing modules (packages)
    # Module = group of functions

#%% 2019-01-23 Functions

# Object oriented programming
    # Make functions to do the work

# Use "def" statement to define a function
# Use "return" to specify what values to return
def testfunction():
    print('in test function')
testfunction()

#%%
def add(a,b):
    c = a+b
    return c

result = add(1,3)
print(result)

#%% 2019-01-23 Johannson Notes Functions

def func0():
    print("test")
func0()

def func1(s):
    """
    Print and string 's' and tell how many characters it has
    """
    
    print(s + " has " + str(len(s)) + " characters")

help(func1)

func1("test")

def square(x):
    """
    Return the square of x
    """
    return x**2

square(2)

def powers(x):
    """
    Return a few powers of x.
    """
    
    return x**2, x**3, x**4

powers(3)
x2,x3,x4 = powers(3)
print(x3)

def powers2(x,y):
    return x**(y), x**(y+1), x**(y+2)

powers2(2,2)

def myfunc(x, p=2, debug=False):
    if debug:
        print("evaluating myfunc for x = " + str(x) + " using exponent p = " + str(p))
    return x**p

myfunc(5)
myfunc(5, debug=True)
myfunc(5, 3, True)

# If you explicitly list names of arguments in functions, the order doesn't matter
myfunc(p = 3, debug=True, x = 4)

f1 = lambda x: x**2
def f2(x):
    return x**2
f1(2), f2(2)
map(lambda x: x**2, range(-3,4))
list(map(lambda x: x**2, range(-3,4)))

list(map(x**2, range(-3,4))) # Doesn't work

#%% 2019-01-23 Modules

%%file mymodule.py
"""
Example of a python module. Contains a variable and a couple functions
"""

my_variable = 0

def my_function1():
    """
    example function
    """
    return my_variable

def my_function2(my_variable, p=2):
    """
    example function 2
    """
    return x**p

#%% Modules continued

import mymodule

help(mymodule)

mymodule.my_function1()

#%% Challenge question 1a

def func_mean1(list):
    return sum(list)/len(list)

#%% Challenge question 1b
    
def func_mean2(list):
    total = 0
    for value in list:
        total += value
    for value2 in list:
        count = 0
        count += value2
    return total/count
#%%
a = [1,2,3,4,5]

func_mean1(a)
func_mean2(a)

#%% Challenge question 2

def func_largest(x,y,z):
    if x > y and x > z:
        print(x)
    elif y > x and y > z:
        print(y)
    else:
        print(z)
        
func_largest(2,4,6)
func_largest(4,2,6)
func_largest(6,4,2)
func_largest(6,2,4)
func_largest(2,6,4)
func_largest(4,6,2)

#%% Functions notes

# variables in functions only exist within that function
# Can integrate global variables into a function by using "global" 
# Functions can be variables or arguments within variables
