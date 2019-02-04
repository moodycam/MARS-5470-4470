# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

person = input('Enter your name:')


#%%
"""
Exercise 1
"""
def scary():
    person = input('Enter your name: ')
    # Creates an input to place your name.
    age = input('Enter your age: ')
    # Creates an input to place your age.
    old = 2019 - int(age) + 100
    # Gives the ouput of the current year minus input age plus 100 years.
    old = str(old)
    # Converts the variable of 'old' to a string so it can be concatenated
    return('Hello ' + person + ". " + "In " + old + " you will be 100 years old.") 
    # Returns the ouput: Hello [inputted name]. In [outputted year] you will
    # be 100 years old.
    
scary()
#%%
"""
Exercise 2
"""
def divide():
    number = input('Enter your number: ')
    # Creates a prompt to enter a number
    number = int(number)
    # Turns the number into an integer
    m = number % 2
    # Returns the remainder after dividing number by 2
    if m > 0:
        print("Odd.")
        # If "m" is greater than 0 than the number is odd, otherwise it is even.
    else:
        print("Even.")

divide()
#%%
"""
Exercise 9
"""
def guess():
    import random
    a = random.randint(1,9)
    # Chooses a random number between (and including) 1 and 9
    number = int(input('Give me a number between 1 and 9: '))
    # Asks the user to input a number between 1 and 9 as an attempted guess
        # for the random number chosen by the program
    if a > number:
        print("Too Low!")
        # If the generated number is greater than the chosen number.
    elif a < number:
        print("Too High!")
        # If the generated number is lower than the chosen number.
    else:
        print("Perfect!")
        # If you're exactly right.
        
guess()

#%%
"""
Exercies 13
"""
def gen_fib():
    count = int(input("How many fibonacci numbers would you like to generate?"))
    i = 1
    if count == 0:
        fib = []
        # If the input is '0' print nothing
    elif count == 1:
        fib = [1]
        # Otherwise, if the input is '1' print '1'
    elif count == 2:
        fib = [1,1]
        # If the input is '2' print '1, 1'
    elif count > 2:
        fib = [1,1]
        while i < (count - 1):
            fib.append(fib[i] + fib[i-1])
            i += 1
            # Else if the input is greater than '2', first print '1,1' to get
                # started, then take the count and count-1 and add them
                # together to create a new variable 'i'
            # Eventual return is 'fib', which is the final elif return if the
                # input is greater than '2'
    return fib

gen_fib()
#%%
"""
Dr. Harrison's Answer for Exercise 13
"""
fib1 = 0; fib2 = 1;

counter = 5

print(fib2)
while counter>1:
    print(fib2+fib1)
    temp = fib2 + fib1
    fib1 = fib2
    fib2 = temp
    counter -= 1
    
"""
Changing Dr. Harrison's code into a function
"""    
def gen_fib2():
    count = int(input("How many fibonacci numbers would you like to generate?"))
    fib1 = 0; fib2 = 1
    while count>1:
        print(fib2+fib1)
        temp = fib2 + fib1
        fib1 = fib2
        fib2 = temp
        count -= 1
    return

gen_fib2()
#%%
"""
Exercise 25 Working (First Attempt)
"""
def big_guess():
    import random
    Min = 0
    Max = 100
    num = random.randint(Min, Max)
    Try = 0
    Running = True
    Answer = None
    while Running:
        print("Is it %s?" % str(num))
        Answer = input()
        if "no" in Answer.lower() and "lower" in Answer.lower():
            num -= random.randint(1,4)
        elif "no" in Answer.lower() and "higher" in Answer.lower():
            num += random.randint(1,4)
        elif Answer.lower() == "no":
            print("Higher or lower?")
            Answer = input()
        if Answer.lower() == "higher":
            num += random.randint(1, 4)
        elif Answer.lower() == "lower":
            num -= random.randint(1, 4)
        elif Answer.lower() == "yes":
            if Try < 2:
                print("Yes! It only took me " + str(Try) + " tries!")
            elif Try < 2 and Try < 10:
                print("Pretty well for a program, " + str(Try) + " tries.")
            else:
                print("Fuck I suck, it took " + str(Try) + "tries.")
        Running = False
    Try += 1
    
    return

big_guess()
#%%
"""
Exercise 25 Working (Second Attempt)
"""
def guess2():
    import random
    Min = 0
    Max = 100
    num = random.randint(Min, Max)
    Try = 0
    Running = True
    Answer = None
    while Running:
        print("Is it %s?" % str(num))
        Answer = input()
        if "no" in Answer.lower() and "lower" in Answer.lower():
            num -= random.randint(1,4)
        elif "no" in Answer.lower() and "higher" in Answer.lower():
            num += random.randint(1,4)
        elif Answer.lower() == "no":
            print("Higher or lower?")
            Answer = input()
        if Answer.lower() == "higher":
            num += random.randint(1, 4)
        elif Answer.lower() == "lower":
            num -= random.randint(1, 4)
        elif Answer.lower() == "yes":
            if Try < 2:
                print("Yes! It only took 1 try!")
            elif Try < 2 and Try < 10:
                print("Decent, " + str(Try) + " tries.")
            else:
                print("Fuck I suck, it took " + str(Try) + "tries.")
        Running = False
    Try += 1
    
    return

guess2()

#%%
"""
Exercise 25 Working (Third Attempt)
"""
def guess3():
    Min = 0
    Max = 100
    Mid = 50
    counter = 1
    print("Please guess a number.")
    condition = input("Is your number " + str(Mid) + "?")
    while condition != "yes":
        counter += 1
        if condition == "higher":
            Min = Mid + 1
        elif condition == "lower":
            Min = Mid - 1
        Mid = (Min + Max) / 2
        condition = input("Is your guess " + str(Mid) + "?")
        ans = "It took" + counter + "times to get your number."
    return ans

guess3()

#%%
"""
Exercise 25 Working (Fourth Attempt)
"""
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
    count += 1
    return ("I solved it in " + count + " tries!")

guess1()