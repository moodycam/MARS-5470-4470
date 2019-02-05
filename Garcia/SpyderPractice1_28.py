# Make a function that finds the mean of a list
# add the elements together
# divide by that number
# x is my list
def mean1(x):
    return sum(x)/len(x)


mylist = [2, 5, 16, 8]
mean1(mylist)

#%%

def mean2(x):
    if (type(mylist[0]) == (int):
        print('Valid') 
    else:
        print('Invalid')
    count = 0.0; mysum=0.0
    for el in x:
        # keep a running sum
        mysum += el
        # track how many elements
        count +=1
    return mysum/count

#%%
    mean2(mylist2)

#%%
mylist2 = ['a',3,'15']


mean2(mylist)

#%%
mylist3 = [1,2,3]
mean2(mylist3)

#%%

def mean3(x):

    count = 0.0; mysum=0.0
    for el in x:
        if (type(el) == (str)):
            int(el)
        # keep a running sum
        mysum += el
        # track how many elements
        count +=1
    return mysum/count

#%%
    mylist4 = ['5', '7', '14']
    mean3(mylist4)
    
#%%
listy = [1,2,3]
for el in listy:
    print(el)
    
for i in [0,1,2]:
    print(listy[i])
#%%
    list2 = ['red', 'blue', 'green']
    print(list2[2])

    for color in list2:
        print(color)

#%%
list5 = ['1','4','8']
mean3(list5)

#%%

def mean3(x):

    count = 0.0; mysum=0.0
    for el in x:
        print('element of x is: ' + str(el))
        if (str(type(el[0:2])) == ("str")) or (str(type(el) == float)):
            int(el)
        # keep a running sum
        mysum += el
        # track how many elements
        count +=1
    return mysum/count

#%%

def mean4(x):
    
    count = 0.0;  mysum=0.0
    for el in x:
        #insert check of elements here
        #check if element is int or float
        #if it is then continue
        #print('element of x is:'  + str(el))
        if type(el) == str:
            el = float(el)
        elif (type(el) == int) or (type)el)==float):
        #keep a running sum
        #count = count + 1
    else:
        print
        
        
#%%
def max3(a,b,c):
    #check if a is biggest
        #if yes return a
    # check if b is bigger than c
        # if yes return b
    # else return c
    if (a > b) and (a>c):
        return a
    elif (b>c):
        return b
    else:
        return c
list7 = [5,4,3]

#%% 
max3(5,4,3)

#%%
person = input('enter your name: ')
person

#WHEN USERS INPUT NUMBER CHANGE THEIR STRING RESPONSE INTO AN INTEGER ONE
#%%
person = input('enter your name: ')
# input for name
age = (input('enter your age: '))

# input for age while being transformed to integer
 def centennial(x):
     return ((100 - age) + 2019)
print ('It will be ' + (centennial()) + ' when you are 100 years old.')

     
     
#%%

#%%
person = str(input('enter your name: '))
# input for name
age = int(input('enter your age: '))
# makes the input for age an int
year = (100 - age + 2019)
# an equation to evalutate centennial year
print ('It will be ' + str(year) + ' when you are 100 years old.')
# print our stuff out
