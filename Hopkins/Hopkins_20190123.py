# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
#%%
# Challenge assignment #1:
## Write a function that takes the mean of a list of values

# As a function 
def mean_a(x):
    a = sum(x)
        #'sum' sums the total string of numbers
    b = len(x)
        #'len' is an inbuilt function that provides an integer that informs
           #the number of integers in a string
    c = a / b
    return c
        
mean_a([1,2,3,4])

# As a function that does not use 'sum'
def mean_b(y):
    a = 0
        # This sets the initial value of the loop as '0' so it has no effect
        # on the sum function that follows
    for n in y:
            # creates a for loop
        a += n
            # c += a is equivalant to c = c + a
            # It adds right operand to the left operand and assign the result
                # the left operand
            # What its doing in the string below is taking the first value of
                # the first number (1) and adding it to 'a' (0) to make 1.
                # Then adding the next number in the string (2) to the sum of
                # the new 'a' (1) to make 3 and so on.
        print(a)
            # prints the result of the final 'a'
    b = len(y)
    c = a / b
    return c

mean_b([1,2,3,4])

# As a function that does not use 'sum' or 'len'
def mean_c(z):
    a = 0
    for n in z:
        a += n
        print(a)
    b = 0
    for n in z:
        b = b+1
            # This creates a count of the string as it starts from '0'
                # (because that's what we set 'b' as) then adds '1' for each
                # number in the string until it runs out of numbers. This
                # eventually creates a count of how many articles are in the
                # string, providing you with a rudimentary 'count.'
            # If we had started 'b' as anything else it would have produced
                # the wrong number for a count.
            # For example, if we set 'b = 5' then the number string that 
                # is created will begin counting at 5 and add 1, therefore
                # starting at '6' in the number string created.
        print(b)
    c = a / b
    return c

mean_c([4,3,2,1])

# Challenge assignment #2
## Write a function that inputs three values and returns the largest
### Don't use the max function
def largest(x):
    b = 0
    for n in x:
        if n > b:
                # If the number in the string is greater than 'b' (which is
                    # set to '0' for the first term) then move on to the next
                    # step. This will continue to repeat through the string
                    # replacing 'b' with the next value if it was greater than
                    # the old 'b'. It will continue to run until it strikes a
                    # number that is less than the number before, and will
                    # stop there.
            b = n
                # This sets 'b' as the final result of the above line.
    print(b)

d = (1,3,2)
largest(d)

#%%
Work Assignment: jrjohansson
Functions
#%% Function 0
def func0():
    print("test")
func0()

# syntax to define a function
    #the 'def' keyword
    #followed by the function's name
        #arguments of the function are given between parentheses followed by a
        #colon.
    #then the function body
    #return object for optionally returning values

#%% Function 1
def func1(s):
   # print a string 's' and tell how many character it has
    print(s + "has" + str(len(s)) + "characters")

help(func1)

func1("test")

#%% Square Function
def square(x):
    # return the square of x
    return x ** 2
    # function that returns a value use the 'return'keyword
    # by default, functions return NONE.
square(4)

#%% Powers Function
def powers(x):
    # return a few powers of x
    return x ** 2, x ** 3, x **4
powers(3)
x2, x3, x4 = powers(3)
print(x3)

#%%
Default argument and keyword arguments
#%% myfunc Function
def myfunc(x, p=2, debug=False):
    #If we don't provide a value of the 'debug' argument when calling the
        #function 'myfunc' it defaults to the value provided in the function
        #definition
    if debug:
        print("evaluating myfunc for x = " + str(x) + "using exponent p=" + str(p))
    return x**p
    ##'myfunc' will automatically apply the exponent 2 to the value of x and
        ##you put nothing in for the 'debug' i.e. 'False'
        ##If you put in true then it runs the 'debug' function as well,
        ##allowing to choose a new exponent by giving 'p' a value.
myfunc(5)
myfunc(5, debug=True)
    # evaluating 'myfunc' for x = 5 using exponent p = 2
    #answer will be same as if ran normally because the default
        #exponent = 2
myfunc(p=3, debug=True, x=7)
    # evaluating 'myfunc' for x=7 using exponent p=3
    #changes x to 7 and p to 3 instead of 2
#%% Unnamed Functions (lambda function)
# In Python we can also create unnamed functions using the 'lambda' keyword
f1 = lambda x: x**2

# is equivalent to

def f2(x):
    return x**2
f1(2), f2(2)

# This technique is useful for example when we want to pass a simple function
# as an argument to another function, like this:

# 'map' is built-in python function
map(lambda x: x**2, range(-3,4))

# in python 3 we can use 'list(...)' to convert the iterator to an explicit
# list
list(map(lambda x: x**2, range(-3,4)))

#%%
Scipy
Functions
#%%
# Functions can optionally return values
def disk_area(radius):
    return 3.14 * radius * radius
disk_area(1.5)

#%%
#There can be Mandatory and Optional parameters within a function

##Mandatory Paramters
def double_it(x):
    #doubles the x value provided when the function is called
    return x*2
double_it(3)
    #works because you provide an x value (3) to exectue the function on
double_it()
    # does NOT work because no x value is defined
    
##Optional Parameters
def triple_it(x=2):
    #triples the x value. If no value is provided, triples the
        #default x value (2)
    return x*3

triple_it(3)
    #replaces x=2 with x=3 because you define an x value (3)
triple_it()
    #works becuase in the original function you designate the default x
        #to be 2, so 2 will automatically be ran if nothing else is entered

#%%
#Keyword Arguments
    #Allow you to specify default values
bigx = 10
def double_it(x=bigx):
    return x*2

bigx = 1e9 # Now it is REALLY big

double_it()

#%%
#Using a mutable type in a keyword argument (and modifying it inside the
    #the function body)
def add_to_dict(args={'a': 1, 'b': 2}):
    for i in args.keys():
        args[i] += 1
    print(args)
