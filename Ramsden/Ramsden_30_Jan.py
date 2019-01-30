#!/usr/bin/env python
# coding: utf-8

# In[8]:


#Problem 13 Fibonacci

fib1 = 0; fib2 = 1;

counter = 5

print(fib2)
while counter>1:
    print('counter = ' + str(counter))
    print('fib1 =' + str(fib1))
    print('fib2 =' + str(fib2))
    #add the previous 2 numbers and print
    print(fib2+fib1)
    #replace numbers for next loop
    #save the new number to go into fib2  
    temp = fib2 + fib1
    #put fib2 into fib1
    fib1 = fib2
    #put new number into fib1
    fib1 = fib2
    # put new number into fib2
    fib2 = temp 
    #counter = counter - 1
    counter -= 1
     


# In[9]:


#Numpy from scipy-lectures.org  they change the website going from google
#doesnt work

#First we will import the packet 


# In[13]:


import numpy as np


# In[14]:


a = np.array([0,1,2,3])


# In[12]:


a


# In[15]:


a


# In[16]:


L = range(1000)


# In[17]:


get_ipython().run_line_magic('timeit', '[i**2 for i in L]')


# In[18]:


a = np.arange(1000)


# In[19]:


get_ipython().run_line_magic('timeit', 'a**2')


# In[23]:


a = np.array([1,2,3])


# In[27]:


b = np.array([1,2,3,4])


# In[29]:


a.dot


# In[30]:


#We did 2D and 3D arrays from the sciypy examples
#the program can tell you information about the array like
#dimension (b.ndim) shape(b.shape) or length (len(b))


# In[38]:


b = np.arange(5,0,-2)


# In[39]:


b


# In[40]:


c = np.arange(-2,-30,-2)


# In[41]:


c


# In[42]:


d = np.linspace(0,1,6)


# In[43]:


d


# In[45]:


x = np.linspace(0,2,4)


# In[46]:


x


# In[47]:


y = np.random.rand(4)


# In[48]:


y


# In[49]:


#Creating arrays using functions


# In[50]:


a = np.ones((3,3))


# In[51]:


a


# In[52]:


b = np.zeros((2,2))


# In[53]:


b


# In[54]:


#Basic data types


# In[55]:


a =np.array([1,2,3])


# In[56]:


a.dtype


# In[57]:


b = np.array([1.,2.,3.])


# In[58]:


b.dtype


# In[ ]:


#Basic Visualization


# In[61]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[62]:


import matplotlib.pyplot as plt


# In[63]:


x = np.linspace(0, 3, 20)
y = np.linspace(0, 9, 20)


# In[64]:


plt.plot(x,y)


# In[65]:


image = np.random.rand(30, 30)
plt.imshow(image, cmap=plt.cm.hot)


# In[68]:


y = np.cos(x)
plt.plot(x,y)
plt.xlabel('x')
plt.ylabel('cos(x)')


# In[69]:


image = np.random.rand(30, 30)
plt.imshow(image, cmap=plt.cm.gray)


# In[70]:


#Indexing and Slicing


# In[71]:


a = np.arange(10)
a


# In[72]:


a[2:9:3] 


# In[73]:


a[:4]


# In[74]:


a = np.arange(10)
a[5:] = 10
a


# In[75]:


b = np.arange(5)
a[5:] = b[::-1]
a


# In[76]:


np.arange(6) + np.arange(0, 51, 10)[:, np.newaxis]


# In[ ]:




