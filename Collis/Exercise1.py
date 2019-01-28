# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 15:30:06 2019

@author: srv_veralab
"""

#Create a program that asks the user to enter their name and their age. 
#Print out a message addressed to them that tells them the year that they will turn 100 years old.
#%%
#Pseudocode:
#First I need to create an variable that asks the user to imput enter their name.

name= input('Please enter your name: ')

#Now create a variable that asks the user for their age. This variable will be of type string.

age= input('Please enter your age: ')

#%%

#Now make a function that determines what year they will turn 100 years old.

#Age-100= the number of years left for the person to turn 100 years old.
#To do math with the variable age we need to convert it into an integer.

y= 100- int(age)
print(y)
#%%
#Once you know how many years it will take them to turn 100, add that number to 2019.

z= y+2019
print(z)
#%%

#Now print a message that tells the user that they will turn 100 in x year.

print(name + ', you will be turning 100 in the year ' + str(z) + '.')

#%%

#Now make that all fit into one cell.

name= input('Please enter your name: ')
age= input('Please enter your age: ')
y= 100- int(age)
z= y+2019
print(name + ', you will be turning 100 in the year ' + str(z) + '.')

#%%

# This is the solution from Practice Python

name = input("What is your name: ") 
age = int(input("How old are you: ")) 
year = str((2014 - age)+100) 
print(name + " will be 100 years old in the year " + year) 

#This gave me 2094. The number I got was 2099.

#%%

# It gave me different numbers because the year in practice python is 2014. It should be 2019.

name = input("What is your name: ") 
age = int(input("How old are you: ")) 
year = str((2019 - age)+100) 
print(name + " will be 100 years old in the year " + year) 





