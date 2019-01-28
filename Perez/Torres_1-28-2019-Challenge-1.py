# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#make a function that finds the mean of a list
#add the elements together
#divide by the number

# x is my list

def mean(x):
    return (sum(x)/len(x))
#%%

mylist =[2, 5, 16, 8]
mean(mylist)
mylist2 = [ 2, 3, 6]
#%%

def mean2(x):
    count = 0.0; mysum = 0.0
    for el in x:
        #insert check elements here
        #check if element is an int or a float
        #if it is then continue
        print('element of x is:' + str(el))
        if (type(el) == int) or (type(el) == float):
            mysum += el
            count += 1
        else:
            print('no')
            return
        #if not then leave the function

    return mysum/count
        #%%
mean2(mylist2)

#%%

def max3(a, b ,c):
    # check if a is biggest
        #if yes return a
    #check if b is bigger than c
        #if yes return b
    #else reutrn c
    if (a>b) and (a>c):
        return a
    elif (b>c):
        return b
    else:
        return c
    #%%
max3(8,44,34)