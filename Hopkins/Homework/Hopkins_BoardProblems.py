# -*- coding: utf-8 -*-
"""
Created on Mon Jan 28 14:13:53 2019

@author: srv_veralab
"""

"""
Function will test the input to make sure its a number
"""
def mean(z):
    a = 0; b = 0  
    for n in z:
        # Creates a 'for loop'
        if (type(n)==int) or (type(n)==float):
            # Before anything else is run, this line of code checks to make
                # sure the input is in face an integer or float, otherswise
                # it will move on to the 'else' statement below.
            print(n)
            # If the element is an integer or a float then it will be printed
                # to be used for the next line.
        else:
            return("Incorrect variable:" + n)
            # If the element was not an integer or float then the print will 
                # provide the green words above.
            # Returns the elements in the output as the element causing the
                # problem.
        a += n
        print(a)
    for n in z:
        b += 1
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
    return a / b
#%%
mean([4,3,2,1])
#%%
mean([4,3,'a',2,1])
#%%
def float1(x):
    for y in x:
        if type(y)==str:
            y = float(y)
        elif (type(y)==int) or (type(y)==float):
            mysum += y
            
