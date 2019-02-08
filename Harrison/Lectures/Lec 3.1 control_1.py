# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

# make a function that finds the mean of a list
# add the elements to gether 
# divide by the number

# x is my list
def mean1(x):
    return sum(x)/len(x)

#%%

mylist = [2, 5, 16, 8]
mean1(mylist)

#%%
# try something easy to test
mylist = [5, 5, 6, 6]
mean1(mylist)

#%%

listy = [1,2, 3]
for el in listy:
    print(el)

for i in [0,1,2]:
    print(listy[i])

list2 = ['red', 'blue','green']
print(list2[2])

for color in list2:
    print(color)

#%%
# now do it with no built in functions

def mean2(x):
    # initialize my counters
    count = 0.0; mysum=0.0
    
    # loop over the elements of the list
    for el in x:
        # insert check of elements here
        #check if element is an int or a float
        # if it is then continue
        #print('element of x is:' + str(el))
        if type(el) == str:
            el = float(el)
        elif (type(el) == int) or (type(el)==float):
            # keep a running sum
            #mysum = mysum + el
            mysum += el
            # track how many elements
            #count = count + 1
            count += 1
        # if not then leave function
        else:
            print('your list has non-numbers...exiting')
            return
    
    return mysum/count
#%%
mean2(mylist)

#%%
mylist2 = ['a', 3 , '15']

mean2(mylist2)

# modifiy the above funciton to check that the input can 
# be averaged
#%%
mylist3 = ['5', '7', '14']
# modify function above to change the type from str 
# to float
mean2(mylist3)

        
