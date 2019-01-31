# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 14:22:51 2019

@author: tjyamashta_dell
"""

# Johansson Tuturial through loops. Homework for 2/1/2019

#%% Modules
import math

x = math.cos(2 * math.pi)
print(x)

# From a MODULE, import ALL DATA
from math import *
x = cos(2*pi)
print(x)

from math import cos, pi
x = cos(2*pi)
print(x)

import math

print(dir(math))
help(math.log)
# With logs, if base undefined, uses natural log
log(10)
log(10,2)


#%% Variables and types


# Variables are equivalent to objects in R
x = 1.0
my_variable = 12.2

type(x)

x = 1
type(x)

print(y)  # This will print an error because y undefined

b1 = True
b2 = False
type(b1)

x = 1.0 - 1.0j
type(x)
print(x)
print(x.real, x.imag)

import types

x = 1.0
# Logical statement checking that x is a float
type(x) is float
type(x) is int
# isinstance can also be used 
isinstance(x, float)

# Can print a variable and its type
x = 1.5
print(x, type(x))
x = int(x)
print(x, type(x))
z = complex(x)
print(z, type(z))
x = float(z)
# Complex variables cannot be type casted to other variable types
y = bool(z.real)
print(z.real, "->", y , type(y))
y = bool(z.imag)
print(z.imag, "->", y, type(y))

"""
Types of variables: 
    float = number with decimals
    int = number without decimals
    bool = boolean. Logical statement
    complex = comlex numbers. Numbers with imaginary numbers
"""

#%% Operators and comparisons

# Most arithatic operators work as expected
# Can run multiple arguments in the same line using a comma between functions
1+2, 1-2, 1*2, 1/2
1.0 + 2.0, 1.0-2.0, 1.0*2.0, 1.0/2.0

# In previous version of python, to force integers to divide as floats, needed to use //
# POWER operator is ** not ^
2**2

# Boolean operators are spelled out as words
True and False

not False

True or False

# Comparison operators 
# >, <, >=, <=, ==, is 

2 > 1, 2 < 1
2 > 2, 2 < 2
2 >= 2, 2 <= 2
[1,2] == [1,2]
l1 = l2 = [1,2]
l1 is l2

#%% Compound types

s = "Hello world"
type(s)

len(s)

s2 = s.replace("world", "test")
print(s2)

s[0]

s[0:5]
s[6:]
s[:]
s[::2]

# Can concatenate strings using print statement and commas between values
# Can also C-style formatting for strings

#%% Lists

# Lists are another data type that we have covered already multiple times
l = [1,2,3,4,5]
print(type(l))
print(l)

# Can do indexing on lists

# Can make a list of a range
# Can convert a string as a list by typecasting it to the list data type
# Can sort lists using the sort function: variable.sort()

# Can use the append function to add items to lists: variable.append("value")

"""
I am skipping over the actually coding part of this tutorial because we have already done it before
See Running_Code.py sections for code related to these things
"""

#%% Tuples

# Tuples are non-modifiable lists

#%% Dictionaries

# Dictionaries are like lists but each element has a key-value pair
# There is a reference associated with a value

params = {"parameter1" : 1.0,
          "parameter2" : 2.0,
          "parameter3" : 3.0,}

print(type(params))
print(params)
print("parameter1 = " + str(params["parameter1"]))
# Can modify dictionaries the same way as lists

#%% Control flow

"""
If Elif Else statements were covered in previous things. See code in Running_Code.py
For loops also covered and available in Running_code.py
While loops are the same as above
"""

