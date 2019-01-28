# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:14:23 2019

@author: zti318
"""

# make a function that takes the max of three arguments

def max3(a,b,c):
    # pseudocode:
    # check if a is biggest
        # if yes return a 
    # check if b is bigger than c
        # if yes return b
    # else return c
    
    if (a > b) and (a>c):
        return a
    elif (b>c):
        return b
    else:
        return c
    
#%%
        
max3(15, 29, 3.2)