#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:16:19 2019

@author: roryeggleston
"""

def mean1(l):
    myMean = float(sum(l) / len(l))
    return myMean
#%%
l = [1, 2, 3, 4]
#%%
mean1(l)
#%%
def NotSum(l):
    total = 0
    for val in l:
        total += val
    return total
#%%
def mean2(l):
    myMean2 = float(NotSum(l) / len(l))
    return myMean2
#%%
NotSum(l)
#%%
mean2(l)
#%%
mylist2 = ['a', 3, '15']
#%%
def mean4(x):
    count = 0.0; mysum=0.0
    for el in x:
        #insert check of elements here
        #check if element is an integer or a float
        #if it is then continue
        #print('element of x is':' + str(el))
        
        print('element of x is:' + str(el))
        if (type(el) == int) or (type(el) == float):
            #keep a running sum
            #mysum = mysum + el
            mysum += el
            #track how many elements
            count += 1
        #if not then leave function
        else:
            print('your list has non-numbers...exiting')
            return

    return mysum/count
#%%
mean4(l)
#%%
list3 = [5, 5, 6, 6]
#%%
mean2(list3)
#%%
def NotSum(l):
    total = 0
    for val in l:
        total += val
    return total
##%
def NotLen(l):
    ListLen = 0
    for i in l:
        ListLen += 1
    return ListLen
#%%
def mean3(l):
    myMean3 = float(NotSum(l) / NotLen(l))
    return myMean3
#%%
mean3(list3)
#%%
def largest(l):
    myMax = float('-inf')
    for i in l:
        print (i)
        print (i > myMax)
        if i > myMax:
            myMax = i
    return myMax
#%%
def largest2(a,b,c):
    #pseudocode:
    #check if a is biggest
        #if yes return a
    #chck if b is bigger than c
        #if yes return b
    #else return c
    
    if (a > b) and (a > c):
        return a
    elif (b > c):
        return b
    else:
        return c
#%%
largest2(23, 1101, 14)
list4 = [234, 345,456, 1204578]
l = [1, 5, 6, 4]
#%%
largest(list4)
#%%
#Ask for name
#Ask for age
#Add 2019 and (100- age)
#Output the answer from above
name = input('Enter your name: ')
age = int(input('What is your age?: '))
year100 = str(2019+(100-age))
print(name + " will be 100 years old in the year " + year100)
#you can't add a string and a list, it will say SCREW YOU if you do
#%%
#Ask for a number
#Determine whether number is odd or even
#return "Even numbers always produce even numbers when multiplied" or "Odd numbers produce decimal numbers when divided" depending on the above answer
number = int(input('Enter a number: '))
#this required you to include int before the input bc otherwise it can't tell how to handle the input number
EvenOdd = number % 2
if EvenOdd > 0:
    print ("Odd numbers produce decimal numbers when divided")
else:
    print ("Even numbers always produce even numbers when multiplied")
#%%
#generate a random number
#ask user to guess and input a number
#tell them "Your number was too low", "Your number was too high", or "Your number matches my number!"
#Keep this going until they input "Exit" and when they do, print out how many guesses they've done to get the right one
import random
GuessNumber = random.randint(1,9)
guess = 0
count = 0
while guess != GuessNumber and guess != "Exit":
    guess = input("What is your guess?: ")
    if guess == "Exit":
        break
    guess = int(guess)
    count += 1
    if guess < GuessNumber:
        print("Your number was too low.")
    elif guess > GuessNumber:
        print("Your number was too high.")
    else:
        print("Your number matches my number!")
        print("And it only took you ",count," guesses!")
# MAKE SURE TO CHECK YOU HAVE ALL QUOTES, APOSTROPHES, AND PARENTHESES/BRACKETS OR SURPRISE SURPRISE IT WON'T WORK YA BIG NUMBSKULL
#%%
#Ask for the number of Fibonnaci numbers desired
#Generate the desired number of Fibonnaci numbers

def generate_fib():
    FibNumber = int(input("How many Fibonnaci numbers do you want?: "))
    i = 1
    if FibNumber == 0:
        fi = []
        
    elif FibNumber == 1:
        fi = [1]
    elif FibNumber == 2:
        fi = [1,1]
    elif FibNumber > 2:
        fi = [1,1]
        while i < (FibNumber - 1):
            fi.append(fi[i] + fi[i-1])
            #append adds a selected thing onto the end of a list, so combined with the += thing, you get the Fib iterations of the desired length
            i += 1
    return fi
print (generate_fib())
#STOP FORGETTING TO INCLUDE PRINT YOU TWIT
#%%
#ask user to think of a number between 0 and 100
#make the program spit out a number within that range, and ask if it is correct
#allow the user to type in "Too High", "Too Low", or "Correct"
#If the answer was "Too High", make the program guess lower; if the answer was "Too Low", make the program guess higher; if it was "Correct", make the program spit out "Thank you for playing" and end the game
#Repeat the above two steps as many times as is necessary until "Correct" is obtained from the user
import random

MIN = 0
MAX = 100
NUMBER = random.randint(MIN, MAX)
#this bit is the thing that will "guess" a number
TRY = 0
#this keeps track of the number of attempts
RUNNING = True
#this just asks whether the game is still going, if the answer was correct, it will switch to False
ANSWER = None
#Keeps track of the current answer from the user (correct, too low, etc)

while RUNNING:
    print ("Is %s too high, too low, or correct?" % str(NUMBER))
    ANSWER = input()
    #input is whatever the user says (correct, too high, etc)
    if "no" in ANSWER.lower() and "too high" in ANSWER.lower():
        NUMBER -= random.randint(1, 4)
        #if the answer given by the user includes "no" and "too high" then the program will guess between 1 and 4 integers lower
    elif "no" in ANSWER.lower() and "too low" in ANSWER.lower():
        NUMBER += random.randint(1, 4)
        #if the answer given by the user include "no" and "too low" then the program will guess between 1 and 4 integers higher
    elif ANSWER.lower() == "no":
        print ("Higher or lower?")
        #if the user simply says "no", the program will ask "Higher or lower?"
        Answer = input()
        if ANSWER.lower() == "higher":
            NUMBER += random.randint(1, 4)
            #if the user answers "higher" then the program will guess between 1 and 4 integers higher
        elif ANSWER.lower() == "lower":
            NUMBER -= random.randint(1, 4)
            #if the user answers "lower" then the program will guess between 1 and 4 integers lower
    elif ANSWER.lower() == "correct":
        if TRY < 2:
            print ("Hooray, I did it in %s try!" % str(TRY))
            #if the user answers "correct", the program will tally the tries and if they are less than two, will return this message
        elif TRY < 2 and TRY < 5:
            print ("It took me %s tries, not too bad, right?" % str(TRY))
            #if the user answers "correct", the program will tally the tries and if they are greater than 2 but less than 5 it will return this message
        else:
            print ("Awwww, that took me %s tries" % str(TRY))
            #if the user answers "correct", the program will tally the tries and if they are more than 5 it will return this message
        RUNNING = False
    TRY += 1
print ("Thanks for playing this game with me!")
