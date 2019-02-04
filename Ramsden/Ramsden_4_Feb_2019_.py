#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np


# In[2]:


import matplotlib.pyplot as plt


# In[3]:


a = np.arange(10)


# In[4]:


a


# In[11]:


plt.plot(a,3*a + 5)
plt.xlabel('a')
plt.title('Plot')
plt.xlabel('a + 5')
plt.show()


# In[10]:


#We are revieweing the Numpy array plus plots


# In[12]:


b = np.zeros((5,6))


# In[13]:


b


# In[14]:


b[0,1,] = 10


# In[15]:


b


# In[16]:


plt.pcolormesh(b)


# In[17]:


b[1,0,] = 10


# In[18]:


plt.pcolormesh(b)


# In[19]:


b[2,2,] = 10


# In[20]:


plt.pcolormesh(b)


# In[33]:


d = np.random.randn(1000)


# In[ ]:


#randn to make them evenly distributed


# In[34]:


d


# In[35]:


d.mean()


# In[36]:


d.max()


# In[37]:


d.min()


# In[38]:


plt.hist(d)


# In[39]:


#Class of February 4th
#Copies and views


# In[40]:


a = np.arange(10)
a

b = a[::2]
b

np.may_share_memory(a, b)

b[0] = 12
b

a   # (!)


a = np.arange(10)
c = a[::2].copy()  # force a copy
c[0] = 12
a


np.may_share_memory(a, c)


# In[41]:


#np.may_share_nenory() is used to check if two arrays share the same memory
#block. 


# In[42]:


is_prime = np.ones((100,), dtype=bool)


# In[43]:


is_prime


# In[44]:


is_prime[:2] = 0


# In[45]:


is_prime


# In[46]:


N_max = int(np.sqrt(len(is_prime) - 1))
for j in range(2, N_max + 1):
    is_prime[2*j::j] = False


# In[47]:


is_prime


# In[48]:


#FANCY INDEXING USING BOOLEAN MASKS


# In[49]:


np.random.seed(3)
a = np.random.randint(0, 21, 15)
a

(a % 3 == 0)


mask = (a % 3 == 0)
extract_from_a = a[mask] # or,  a[a%3==0]
extract_from_a           # extract a sub-array with the mask


# In[50]:


#iNDEXING WITH A MASK CAN BE VERY USEFUL TO ASSIGN A NEW VALUE TO A SUB-ARRAY


# In[51]:


a[a % 3 == 0] = -1
a


# In[52]:


#INDEXING WITG AN ARRAY OF INTEGERS


# In[53]:


a = np.arange(0, 100, 10)
a


# In[54]:


#Indexing can be done with an array of integers, where the same index is repeated several time:


# In[55]:


a[[2, 3, 2, 4, 2]]  # note: [2, 3, 2, 4, 2] is a Python list


# In[57]:


#New values can be assigned with this kind of indexing:


# In[58]:


a[[9, 7]] = -100
a


# In[59]:


#When a new array is created by indexing with an array of integers, the new array has the same shape as the array of integers:


# In[63]:


a = np.arange(10)
idx = np.array([[3,4], [9,7]])
idx.shape


# In[64]:


a[idx]


# In[66]:


#1.3.2. Numerical operations on arrays
#Elementwise operations


# In[67]:


#Basic operations
#with scalars


# In[68]:


a = np.array([1, 2, 3, 4])
a + 1

2**a


# In[69]:


#All arithmetic operates elementwise


# In[70]:


b = np.ones(4) + 1
a - b

a * b


j = np.arange(5)
2**(j + 1) - j


# In[71]:


a = np.arange(10000)
get_ipython().run_line_magic('timeit', 'a + 1')

l = range(10000)
get_ipython().run_line_magic('timeit', '[i+1 for i in l]')


# In[72]:


#OTHER OPERATIONS


# In[74]:


a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
a == b


# In[75]:


a > b


# In[76]:


#These last operations were comparisons between 2 arrays (a and b)


# In[77]:


a = np.array([1, 2, 3, 4])
b = np.array([4, 2, 2, 4])
c = np.array([1, 2, 3, 4])
np.array_equal(a, b)


# In[78]:


np.array_equal(a, c)


# In[79]:


# It can also do pair wise comparisons


# In[80]:


#Logical Operations


# In[81]:


a = np.array([1, 1, 0, 0], dtype=bool)
b = np.array([1, 0, 1, 0], dtype=bool)
np.logical_or(a, b)


# In[82]:


np.logical_and(a, b)


# In[83]:


#Transcendental functions


# In[84]:


a = np.arange(5)
np.sin(a)


# In[85]:


np.log(a)


# In[86]:


#Shape mismatches


# In[87]:


a = np.arange(4)
a + np.array([1, 2]) 


# In[88]:


a = np.triu(np.ones((3, 3)), 1)   # see help(np.triu)
a


# In[89]:


a.T


# In[90]:


#Basic Reductions


# In[98]:


#Computing sums


# In[100]:


x = np.array([1, 2, 3, 4])
np.sum(x)


# In[101]:


x.sum()


# In[102]:


#Sum by rows and by columns 


# In[103]:


x = np.array([[1, 1], [2, 2]])
x


# In[104]:


x.sum(axis=0)


# In[105]:


x[:, 0].sum(), x[:, 1].sum()


# In[106]:


x.sum(axis=1) 


# In[107]:


x[0, :].sum(), x[1, :].sum()


# In[108]:


#Same idea in higher dimensions 


# In[109]:


x = np.random.rand(2, 2, 2)
x.sum(axis=2)[0, 1]  


# In[110]:


x[0, 1, :].sum()


# In[111]:


#Other reductions


# In[112]:


x = np.array([1, 3, 2])
x.min()


# In[113]:


x.max()


# In[114]:


x.argmin() 


# In[115]:


#Index of the minimum


# In[116]:


x.argmax()


# In[117]:


#Index of the maximum


# In[118]:


#Logical Operations


# In[119]:


np.all([True, True, False])


# In[120]:


np.any([True, True, False])


# In[121]:


#Statistics


# In[122]:


x = np.array([1, 2, 3, 1])
y = np.array([[1, 2, 3], [5, 6, 1]])
x.mean()


# In[123]:


np.median(x)


# In[124]:


np.median(y, axis=-1)


# In[125]:


x.std()


# In[126]:


#Example data


# In[127]:


filen = 'C:\\Users\\srv_veralab\\Downloads\\populations.txt'


# In[128]:


data = np.loadtxt(filen)


# In[129]:


data


# In[132]:


from matplotlib import pyplot as plt


# In[134]:


year, hares, lynxes, carrots = data.T


# In[135]:


plt.axes([0.2, 0.1, 0.5, 0.8]) 
plt.plot(year, hares, year, lynxes, year, carrots) 
plt.legend(('Hare', 'Lynx', 'Carrot'), loc=(1.05, 0.5)) 


# In[136]:


populations = data[:, 1:]
populations.mean(axis=0)


# In[137]:


populations.std(axis=0)


# In[138]:


np.argmax(populations, axis=1)


# In[ ]:




