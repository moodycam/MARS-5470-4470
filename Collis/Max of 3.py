# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:18:43 2019

@author: srv_veralab
"""

# Make a function that takes the max of three arguments

def max3(a,b,c):
    #Pseudocode:
    #check if a is biggest
        #if yes return a
    #check if b is greater than c
        #if yes return b
    #else return c
    
    if (a>b) and (a>c):
        return a
    elif (b>c):
        return b
    else:
        return c
    
#%%   
max3(1,2,3)