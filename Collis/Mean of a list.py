# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Make a function that finds the mean of a list
#Add the elements together
#divide by the number

#x is my list

def mean(x):
    return sum(x)/len(x)

myList=[5, 5, 6, 6]
mean(myList)

#you might need to highlight the whole fucntion and then press run to execute

#Now I will try to do it without any built in functions

def mean2(x):
    count=0.0; mySum=0.0
    for el in x:
        mySum += el
        count += 1
    return mySum/count

mean2(myList)

#Now we want to check if the elements are an int or a float
#%%
def mean3(x):
    count=0.0; mySum=0.0
    for el in x:
        if (type(el[0])== int) or (type(x[0]== float))
            mySum += el
            count += 1
        return mySum/count
        else: 
            print('your list has non-numbers')
#%%
mean2(myList)
#%%
myList1=[5, 7, 14, 'alice']
#%%
mean2(myList1)
#%%
def mean4(x):
    count=0.0; mySum=0.0
    for el in x:
        print('Element: ' + str(el))
        if (type(el)== int) or (type(el)== float):
            mySum += el
            count += 1
            print(mySum)
            print(count)
        else: 
            print('your list has non-numbers')
            return
        
        return mySum/count

#%%
mean4(myList1)
                     
            
            
            
            
            