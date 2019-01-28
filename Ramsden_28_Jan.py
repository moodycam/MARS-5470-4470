# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Esta es la clase del Lunes 28 de Enero del 2019


#%%

#The first thing you need to do if you havent done it, is to create a Github 
#account www.github.com

#During class, we reviewed the challenge question from last wednesday
#mean of 3 numbers without using sum function and max of a list of 3 numbers
#without using the max function.

#First we started reviweing the problems 

#

#%%

#Mean of a given list

x = [2,4,6,8,10]

print (x)


#%%

# X is my list

def mean(x):
    return sum(x)/len(x)

mean (x)

#%%

#First we defined the list of x and then we define the function in this case 
#"mean" and it will give us what we define it to do in this case sum x and 
#divide it by its length 

#%%

#No we will do without using the sum function

#%%

# el = element ,  so what it does, it starts the count in 0 and for each element
#in x it will do mysum + el  its basically doing the sum function and it 
#will track how many elements it has

def mean2(x):
    count = 0.0; mysum=0.0
    for el in x:
        mysum += el
        count += 1
    return mysum/count
        #%%

x = [2,4,6,8,10]
#%%

mean2 (x)

#THis gives us the average using the function mean2

#%%

#Now we will use the same function but we will tell us if we have a letter instead of numbers 
#to print an error message 


#check if numbers before the loop/in the loop

#if list is a float 
#%%
mylist2 = ['a','3','15']

type(mylist2)

#In this list I did a mistake, since i put the numbers in between quotes
#it takes the numbers as if they were a string and not an int or float


#%%
mylist2 = ['a',3,15]

def mean2(x):
    
    count = 0.0; mysum=0.0
    
    for el in x:
        
        if (type(el) == int) or (type(el) == float):
            print ('here') 
            
        else:
            print('your list has non-numbers')
            return
        
        mysum += el
        count += 1
        
        
    return mysum/count

#in this example, you can change the a to a number and then it will give you
    #the average if you leave the a in the mylist2 it will print the error 
    #message

#%%
    
mean2 (mylist2)
        
#%%

#Convert string to integers

mylist3 = ['a',3,15]

def mean2(x):
    
    count = 0.0; mysum=0.0
    
    for el in x:
        
        if (type(el) == int) or (type(el) == float):
            print ('here') 
            
        else:
            print('your list has non-numbers')
                
        
            return
        
        mysum += el
        count += 1
        
        
    return mysum/count

#%%
#this one it didnt work so just check it but dont run it, it was to try to convert
    #a string into a float 
list4 = ('2',3.0,4.2)

def converter(x):
    
    for el in x:
        
        print (float(x))
        
        return
    
converter(list4)
    

#%%

#Second challenge get the maximum number out of 3
#https://www.practicepython.org/exercise/2016/03/27/28-max-of-three.html


def max3(a,b,c):
    
#check if a is biggest
    #if yes return a
#Check if b is bigger than c
    #if yes return b
#else return c
    
    if (a > b ) and (a > c):
        return a
    elif(b>c):
        return b
    else:
        return c
    
#%%
#This one will give us the number 8 because it is the biggest 
        
max3(8,2,3)

#%%

#Input function

person = input('Enter your name:')

#Here in the console you write your name after the 'Enter your name" and Person gets the value of your name

#%%

x = input ("Enter first number: ")

#when you do this, you have to enter a value of X in the console, right side of the screen. 
#if you dont do it, it wont let you run any more scripts, call me when you are doing this. 
#%%

y = input ("Enter second number: ")

#%%

x + y

#if you add x (1)+ y (2) it will give you 12 instead of 3 because they were defined
#as strings, you have to change them to float in order to be added

float(x) + float(y)

#%%

#now we started this exercise 
#https://www.practicepython.org/exercise/2014/01/29/01-character-input.html

name = input ('Enter your name')

age = input ('Enter your age')


#Trying to substract my age from todays date (year) and add 100 to give me when I will be 100 years old

#When we define age it defines it as string, we need to change it to a float next line
age = float(age)

#now that age is a float it can be added or substracted in the next line, if it is a string it will give you 
#an error 
year = ((2019 - age)+100)

#Now that the math was done, year needs to be changed to string so it can 
#be printed with "name" and "will be 100 years old in the year" which are type strings so we do the next command:
year = str(year)

#In oder words, you cant use the command PRINT with strings floats and integers together, they have to be the 
#same type
print(name + " will be 100 years old in the year " + year  )

#%%

type(name)

type(age)


        
        