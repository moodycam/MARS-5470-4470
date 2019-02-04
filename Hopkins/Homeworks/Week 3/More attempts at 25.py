# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 16:25:43 2019

@author: Miles
"""

# User has a number between 0 and 100

# Computer generates a random number and asks if its right
import random
def guess1():
        ans1 = 1
    while ans1 != "yes":
        guess = random.randint(0, 100)
# Is it right, yes or no?
        ans1 = input("Correct?")
# If yes => done
    if ans1 == "yes":
        print("Great! I solved it!")
# If no
    else:
        # Is it high or low?
        ans2 = input("higher or lower?")
        # If High
        if ans2 == "higher":
            guess += random.randint(1, 5)
            # If Low
        elif ans2 == "lower":
            guess -= random.randint(1, 5)
    return

guess1()
#%%
