# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

#Homework 

#%%

#Problem 1
name = input("What is your name: ")
age = int(input("How old are you: "))
year = str((2014 - age)+100)
print(name + " will be 100 years old in the year " + year)

#%%

#Problem 2

num = input("Enter a number: ")
mod = num % 2
if mod > 0:
    print("You picked an odd number.")
else:
    print("You picked an even number.")
    
#%%

#Problem 9
    
import random

number = random.randint(1,9)
guess = 0
count = 0


while guess != number and guess != "exit":
    guess = input("What's your guess?")
    
    if guess == "exit":
        break
    
    guess = int(guess)
    count += 1
    
    if guess < number:
        print("Too low!")
    elif guess > number:
        print("Too high!")
    else:
        print("You got it!")
        print("And it only took you",count,"tries!")
        
 #%%

#Problem 25
 
MINIMUM = 0
MAXIMUM = 100
NUMBER = random.randint(MINIMUM, MAXIMUM)
TRY = 0
RUNNING = True
ANSWER = None

while RUNNING:
    print "Is it % s?" % str(NUMBER)
    ANSWER = raw_input()
    if "no" in ANSWER.lower() and "lower" in ANSWER.lower():
        NUMBER -= random.randint(1, 4)
    elif "no" in ANSWER.lower() and "higher" in ANSWER.lower():
        NUMBER += random.randint(1, 4)
    elif ANSWER.lower() == "no":
        print "Higher or lower?"
        ANSWER = raw_input()
        if ANSWER.lower() == "higher":
            NUMBER += random.randint(1, 4)
        elif ANSWER.lower() == "lower":
            NUMBER -= random.randint(1, 4)
    elif ANSWER.lower() == "yes":
        if TRY < 2:
            print "Yes! It only took me %s try!" % str(TRY)
        elif TRY < 2 and TRY < 10:
            print "Pretty well for a robot, %s tries." % str(TRY)
        else:
            print "That's so bad, %s tries." % str(TRY)
        RUNNING = False
    TRY += 1
    
print "Thanks for the game!"


#%%

#Johanson loops

for x in [1,2,3]:
    print(x)
    
#%%


for x in range(4): 
    print(x)
#%%

for x in range(-3,3):
    print(x)
#%%
    
for word in ["scientific", "computing", "with", "python"]:
    print(word)
#%%
    
for key, value in params.items():
    print(key + " = " + str(value))
    
#%%
    
for idx, x in enumerate(range(-3,3)):
    print(idx, x)

#%%
    



        