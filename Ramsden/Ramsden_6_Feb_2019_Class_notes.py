#!/usr/bin/env python
# coding: utf-8

# In[1]:


import matplotlib


# In[2]:


import numpy as np


# In[3]:


import matplotlib.pyplot as plt


# In[4]:


get_ipython().run_line_magic('matplotlib', 'inline')


# In[5]:


#Setting up the plot


# In[7]:


plt.subplots(figsize = (2,2))


# In[8]:


#1.3.2


# In[9]:


#Broadcasting


# In[10]:


a = np.tile(np.arange(0, 40, 10), (3, 1)).T
a




b = np.array([0, 1, 2])
a + b


# In[11]:


a = np.ones((4, 5))
a[0] = 2  # we assign an array of dimension 0 to an array of dimension 1
a


# In[12]:


a = np.arange(0, 40, 10)
a.shape

a = a[:, np.newaxis]  # adds a new axis -> 2D array
a.shape

a




a + b


# In[13]:


#Exercise construct an array of distances (in miles) between cities of Route 66: Chicago, Springfield, Saint-Louis, Tulsa, Oklahoma City, Amarillo, Santa Fe, Albuquerque, Flagstaff and Los Angeles.


# In[15]:


mileposts = np.array([0, 198, 303, 736, 871, 1175, 1475, 1544,
       1913, 2448])
distance_array = np.abs(mileposts - mileposts[:, np.newaxis])
distance_array


# In[16]:


x, y = np.arange(5), np.arange(5)[:, np.newaxis]
distance = np.sqrt(x ** 2 + y ** 2)
distance


# In[17]:


plt.pcolor(distance)    
plt.colorbar()    


# In[18]:


#the numpy.ogrid() function allows to directly create vectors x and y of the previous example, with two “significant dimensions”:


# In[19]:


x, y = np.ogrid[0:5, 0:5]
x, y





x.shape, y.shape

distance = np.sqrt(x ** 2 + y ** 2)


# In[20]:


#So, np.ogrid is very useful as soon as we have to handle computations on a grid. On the other hand, np.mgrid directly provides matrices full of indices for cases where we can’t (or don’t want to) benefit from broadcasting:


# In[21]:


x, y = np.mgrid[0:4, 0:4]
x




y


# In[22]:


#Array Shape Manipulation


# In[23]:


#Flattening


# In[24]:


a = np.array([[1, 2, 3], [4, 5, 6]])
a.ravel()

a.T



a.T.ravel()


# In[25]:


#Reshaping, which is the inverse o flattening


# In[26]:


a.shape

b = a.ravel()
b = b.reshape((2, 3))
b


# In[27]:


#or


# In[28]:


a.reshape((2, -1))


# In[29]:


b[0, 0] = 99
a


# In[30]:


#Adding a dimension


# In[31]:


#Indexing with the np.newaxis object allows us to add an axis to an array (you have seen this already above in the broadcasting section):


# In[32]:


z = np.array([1, 2, 3])
z


z[:, np.newaxis]




z[np.newaxis, :]


# In[ ]:


#Dimension Shuffling


# In[33]:


a = np.arange(4*3*2).reshape(4, 3, 2)
a.shape

a[0, 2, 1]

b = a.transpose(1, 2, 0)
b.shape

b[2, 1, 0]


# In[34]:


b[2, 1, 0] = -1
a[0, 2, 1]


# In[35]:


#Resizing


# In[36]:


a = np.arange(4)
a.resize((8,))
a


# In[37]:


b = a
a.resize((4,))   


# In[ ]:


#Sorting Data


# In[38]:


a = np.array([[4, 3, 5], [1, 2, 1]])
b = np.sort(a, axis=1)
b


# In[ ]:


#In-Place Sort


# In[39]:


a.sort(axis=1)
a


# In[ ]:


#Sorting with fancy indexing


# In[40]:


a = np.array([4, 3, 1, 2])
j = np.argsort(a)
j

a[j]


# In[ ]:


#Finding minima and maxima


# In[41]:


a = np.array([4, 3, 1, 2])
j_max = np.argmax(a)
j_min = np.argmin(a)
j_max, j_min


# In[ ]:




