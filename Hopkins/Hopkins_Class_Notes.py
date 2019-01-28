# -*- coding: utf-8 -*-
"""
Created on Wed Jan 23 18:10:11 2019

@author: srv_veralab
"""

#%% Board Notes
Variable Assignments
 myvar = 2
     "myvar" is the variable name
     "=" is the assignment operator
     "2" is what it can be
         Types of assignments (aka "2")
             Integer: 1, 2, 3, 4...
             Float: 2.0, 2.1, 2.2...
             String: 'red', "blue", "123"... (anything in quotes)
             Boolean: TRUE or FALSE
             List: ["cat", "dog", "horse"]
Indexing
    Working with lists
        ex. for exlplanations will be "domestics": ["cat", "dog", "horse"]
    Elements are the items in the list, such as "cat" or "dog"
    The first element in a list is indexed as 0, then 1 and so on
        0 = "cat", 1 = "dog", 2 = "horse"
    To change a value you name the list then [list number] = "new value"
        domestics[1] = "cow" it will change "dog" to "cow"
    domestics[:] prints the whole list
    domestics[:1] will print "dog"
    domestics[:3] would print the whole list
Logical Control
    for, while loops
        for loops are for when you want to go through something quite
        a few times
            ex. 1
                x=[1,2,5,9]
                for y in x:
                    print(y)
        while loops
            ex. 2
                x=3
                while x>0:
                    print(x)
                    x=x-1
            
    if statements:
        if (this statement is true):
            [do this]
            *note, must have indent before [do this]
        ex.
            x=9
            if x<8:
                print("hello")
            else:
                print("goodbye")
Importing Modules
    import math
        Imports the entire math module
    import math.sin
        Just imports a singular function from the module
Object Oriented Programming
    Means "make a function to do the work"
    Pseudo Code:
        def nameoffunction (arguments):
            "def" defines the function
            "nameoffunction" names the function
            "(argument)" prints an argumnet
    ex. 1
        def add(a,b):
            c = a+b
            return c
      #answer 
        result = add(6.3, 8.5)
        print(result)
        add(6.3, 8.5)