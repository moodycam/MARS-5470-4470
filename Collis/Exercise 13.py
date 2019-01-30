# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#%%\

#Problem 13 on Practice Python

#Write a program that asks the user how many Fibonnaci numbers to generate and then generates them. 
#Take this opportunity to think about how you can use functions.
#Make sure to ask the user to enter the number of numbers in the sequence to generate.

#What are Fibonnaci numbers? 
# 0 (invisible)
# 1
# 1
# 1+1=2
# 2+1=3
# 3+2=5
#%%

fib1=0; fib2=1

counter=5

print(fib2)
while counter>0:
    print(fib2+fib1)
    temp= fib2+fib1
    fib1=fib2
    fib2=temp
    counter -= 1
#%%

counter=input('Choose a number: ')
fib1=0; fib2=1 
print(fib2)
while int(counter)>0:
    print('counter= '+ str(counter))
    print('fib1= '+ str(fib1))
    print('fib2= '+ str(fib2))
    print(fib2+fib1)
    temp= fib2+fib1
    fib1=fib2
    fib2=temp
    counter -= 1
    
#