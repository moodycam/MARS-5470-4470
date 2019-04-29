# -*- coding: utf-8 -*-
"""
Created on Mon Apr  8 13:59:09 2019

@author: Miles
"""

#%%
""" Finding Psi (Little trident) """

"""
Find bobcat probability based on vegetation; cover is 'site'
    Run bootstrap analysis (a lot of times, 20 to 200)
        Average all of the results of the bootstrap analysis
"""

"""
Introduction to bootstrap analysis in Python, url:
    https://machinelearningmastery.com/a-gentle-introduction-to-the-bootstrap-method/
"""

"""
Deeper into bootstrap code, url:
    https://pypi.org/project/bootstrapped/
"""
#%%


""" Bootstrap Example Using Bootstrapped Package """
# url: https://pypi.org/project/bootstrapped/#description
#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import scipy.stats as st
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import bootstrapped.compare_functions as bs_comp
#%% Requires the mean and standard deviation of the data
mean = 100
stdev = 10
#%% Setting the population as the data
pop = np.random.normal(loc=mean, scale=stdev, size=50000)
#%% Taking a subset of samples for running bootstrap on
    # Only necessary with very large datasets
# take 1k 'samples' from the larger pop
samples = pop[:1000]
#%% Answer should be 100, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples, stat_func=bs_stats.mean))
#%% Answer should be 10, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples, stat_func=bs_stats.std))
#%%


""" Bootstrap Example Using GitHub Explanation """
# url: https://github.com/facebookincubator/bootstrapped/blob/master/examples/bootstrap_intro.ipynb
#%% Use the same mean, stdev, samples, and pop from above
# Uses all of the same imported packages as above
# Plot the population
count, bins, ignored = plt.hist(pop, 30, normed = True)
plt.show()
#%% Analysis
    # Takes a specific sample size
sample_size = [100, 350, 500, 1000, 2500, 3500, 5000, 8000, 10000]
#%% Unsure of the purpose of this other than to create these variables beforehand
bootstrap_results = []

normal_results = []
#%% 
for i in sample_size:
    samples = np.random.choice(pop, i, replace=True)
    bsr = bs.bootstrap(samples, stat_func=bs_stats.mean, alpha=0.05)
    
    mr = st.t.interval(1-0.05, len(samples)-1, loc=np.mean(samples), scale=st.sem(samples))

    bootstrap_results.append((bsr.lower_bound, bsr.upper_bound))
    normal_results.append(mr)
#%% Creates a plot of the bootstrap results in the upper bound
plt.plot(sample_size, [x[1] for x in bootstrap_results], c='blue')
# Adds to the plot the normal t-distribution in the upper bound
plt.plot(sample_size, [x[1] for x in normal_results], linestyle='--', c='orange')
# Adds to the plot the bootstrap results in the lower bound
plt.plot(sample_size, [x[0] for x in bootstrap_results], c='blue', 
         label='Bootstrap')
# Adds to the plot the normal t-distribution in the upper bound
plt.plot(sample_size, [x[0] for x in normal_results], linestyle='--', c='orange', 
         label='t-distribution')
# Adds the true mean of the data
plt.axhline(pop.mean(), c='black', label='True Mean')
# Adds a legend and title
plt.legend(loc='best')
plt.title('t-distribution vs Bootstrap')
#%%


""" Bootstrap Example Using Second GitHub Explanation """
# url: https://github.com/facebookincubator/bootstrapped/blob/master/examples/bootstrap_ab_testing.ipynb
#%%
# Requires the same packages from the top
#%%
# little a/b test
# score in test are 10% greater than ctrl (per record)
# ctrl has 5x the number of records as test

# 10% lift in test
lift = 1.1
# Test will be used as the "sample"
test = np.random.binomial(100, p=0.2 * lift, size=10000) * 1.0
# Control will be the "pop" dataset
ctrl = np.random.binomial(100, p=0.2, size=50000) * 1.0
#%% Creates a histogram of the Control and Test Data
bins = np.linspace(0, 40, 20)

plt.hist(ctrl, bins=bins, label='Control')
plt.hist(test, bins=bins, label='Test', color='orange')
plt.title('Test/Ctrl Data')
plt.legend()
#%%
# run an a/b test simulation ignoring the lengths of the series (average)
# just what is the 'typical' value
# use percent change to compare test and control
print(bs_comp.percent_change(test.mean(), ctrl.mean()))

print(bs.bootstrap_ab(test, ctrl, bs_stats.mean, bs_comp.percent_change))
#%%
print(len(test))
print(len(ctrl))
#%%
# run an a/b test simulation considering the lengths of the series (sum)
# consider the full 'volume' of values that are passed in
print(bs_comp.percent_change(test.sum(), ctrl.sum()))

print(bs.bootstrap_ab(
    test, 
    ctrl, 
    stat_func=bs_stats.sum,
    compare_func=bs_comp.percent_change
))
#%% Advice from the page
"""
Advice: For most situations we reccomend that we use 'sum' aggregate function
 as this will take the size of the population into account. This can be useful
 if you think your test will increase the total number of individuals in the
 population and not only the typical value per individual.

Exception to the above advice

There are situations where you might need to make some adjustments, for
 example if your holdouts are sized differently.

You give 100 dollars to person_A and you give 1000 dollars to person_B to
 bring people to your store. Say you don't know how many people person_A or
 person_B went after but they did each get paying customers to attend.
 However you do need to correct at the end for the fact that you gave
 person_B 10x more money than person_A
"""
#%%
# Gave $100, got 1k events on the store, $20 per event 
person_A_results = np.random.binomial(100, p=0.2, size=1000) * 1.0

# Gave $1000, got 5k events on the store, $30 per event
person_B_results = np.random.binomial(100, p=0.3, size=5000) * 1.0
#%%
# The test earned much less in terms of total dollars
print(bs.bootstrap_ab(
    person_A_results, 
    person_B_results, 
    stat_func=bs_stats.sum,
    compare_func=bs_comp.difference,
))
#%%
# The test gives ~$10 less per event
print(bs.bootstrap_ab(
    person_A_results, 
    person_B_results, 
    stat_func=bs_stats.mean,
    compare_func=bs_comp.difference,
))
#%%
# If we scale the text by $$ spent - person_A should be a better return on investment
#  - assuming person_A can achieve similar results with 10x more money
print(bs.bootstrap_ab(
    person_A_results, 
    person_B_results, 
    stat_func=bs_stats.sum,
    compare_func=bs_comp.difference,
    scale_test_by=10.,
))
#%%


#%%
""" Code for My Project to Determine Bootstrap Analysis"""
#%% Necessary Packages
import numpy as np
import random as ran
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statistics as st
import scipy.stats as st
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats
import bootstrapped.compare_functions as bs_comp
#%% Inserting the Dataset
# pop = Insert from Excel
file = "C:\\Users\Miles\Downloads\Project_Dataset.xlsx"
data = pd.read_excel(file, sheet_name=4)
#%%
pop2 = np.array(data)
pop1 = pop2.flatten()
pop = pop1.astype(float)
# sample_size = needs to be an exert from the data, or should it just be all the data?
#%%
np.random.sample(500,)
#%% Create blank variables beforehand
bootstrap_results = []
sample_size = [100, 350, 500, 1000, 2500, 3500, 5000, 8000, 10000]
normal_results = []
#%%
data
#%%
pop
#%% Run the bootstrap analysis many times and take the average of the results
for i in sample_size:
    samples = np.random.choice(pop, i, replace=True)
    bsr = bs.bootstrap(samples, stat_func=bs_stats.mean, alpha=0.05)
    
    mr = st.t.interval(1-0.05, len(samples)-1, loc=np.mean(samples), scale=st.sem(samples))

    bootstrap_results.append((bsr.lower_bound, bsr.upper_bound))
    normal_results.append(mr)
#%% Creates a plot of the bootstrap results in the upper bound
plt.plot(sample_size, [x[1] for x in bootstrap_results], c='blue')
# Adds to the plot the normal t-distribution in the upper bound
plt.plot(sample_size, [x[1] for x in normal_results], linestyle='--', c='orange')
# Adds to the plot the bootstrap results in the lower bound
plt.plot(sample_size, [x[0] for x in bootstrap_results], c='blue', 
         label='Bootstrap')
# Adds to the plot the normal t-distribution in the upper bound
plt.plot(sample_size, [x[0] for x in normal_results], linestyle='--', c='orange', 
         label='t-distribution')
# Adds the true mean of the data
plt.axhline(pop.mean(), c='black', label='True Mean')
# Adds a legend and title
plt.legend(loc='best')
plt.title('t-distribution vs Bootstrap')
#%% Requires the mean and standard deviation of the data
mean = np.mean(pop)
stdev = np.std(pop)
#%% Taking a subset of samples for running bootstrap on
    # Only necessary with very large datasets
# take 1k 'samples' from the larger pop
samples = pop[:1000]
#%% Answer should be 100, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples, stat_func=bs_stats.mean))
# mean = 0.033
#%% Answer should be 10, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples, stat_func=bs_stats.std))
# stdev = 0.179
#%%
""" Coding Equations """

"""
1. Break into 2 groups
    Two different equations, one for sites that had at least one capture, (at
    least one 1) and another for sites with just 0's
2. Create functions that conduct equations
    First equation for groups with at least one capture and second equation
    for groups with just 0's
    p variabe at all sites will be constant
        0.5 is recommended by MacKenzie so will be used for now
        Will use temperature as an indicator for future analysis
            Was not measured for this dataset
3. Combine the 2 models
    Combine the two models so that they can be used against the vegetation
    map to create an occupancy model map
"""
#%% Original
def model(x):
    bs_mean = 0.33; i = 0.5; j = 1
    for y in x:
        if (y<1):
            print(i)
        else:
            print(j)
    return bs_mean*i*j

model(samples)
#%%
a = pop
#%% Attempts
n = [0.5 if x == 0 else x for x in a]
#%%
n
#%%
result1 = np.prod(n)
#%%
result2 = bs_mean*result1
#%%
print(result2)
#%%
"""
Code for 0% Canopy Cover
"""
#%% Inserting the Dataset
# pop = Insert from Excel
file = "C:\\Users\Miles\Downloads\Project_Dataset.xlsx"
data0 = pd.read_excel(file, sheet_name=5)
#%%
pop2 = np.array(data0)
pop1 = pop2.flatten()
pop0 = pop1.astype(float)
# sample_size = needs to be an exert from the data, or should it just be all the data?
#%% Requires the mean and standard deviation of the data
mean0 = np.mean(pop0)
#%%
print(mean0)
#%% Taking a subset of samples for running bootstrap on
    # Only necessary with very large datasets
# take 1k 'samples' from the larger pop
samples0 = pop0[:1000]
#%% Answer should be 100, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples0, stat_func=bs_stats.mean))
#%% Important Variables
bs_mean0 = bs.bootstrap(samples0, stat_func=bs_stats.mean)
#%% Important Null Variables
n0 = [0.5 if x == 0 else x for x in samples0]
r01 = np.prod(n0)
r02 = bs_mean0*r01
print(r02)
#%%
"""
Code for 10% Canopy Cover
"""
#%% Inserting the Dataset
# pop = Insert from Excel
file = "C:\\Users\Miles\Downloads\Project_Dataset.xlsx"
data10 = pd.read_excel(file, sheet_name=6)
#%%
pop2 = np.array(data10)
pop1 = pop2.flatten()
pop10 = pop1.astype(float)
# sample_size = needs to be an exert from the data, or should it just be all the data?
#%% Requires the mean and standard deviation of the data
mean10 = np.mean(pop10)
stdev10 = np.std(pop10)
#%%
print(mean10)
#%% Taking a subset of samples for running bootstrap on
    # Only necessary with very large datasets
# take 1k 'samples' from the larger pop
samples10 = pop10[:1000]
#%% Answer should be 100, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples10, stat_func=bs_stats.mean))
#%% Important Variables
bs_mean10 = bs.bootstrap(samples10, stat_func=bs_stats.mean)
#%% Important Null Variables
n10 = [0.5 if x == 0 else x for x in samples10]
r101 = np.prod(n10)
r102 = bs_mean10*r101
print(r102)
#%%
"""
Code for 30% Canopy Cover
"""
#%% Inserting the Dataset
# pop = Insert from Excel
file = "C:\\Users\Miles\Downloads\Project_Dataset.xlsx"
data30 = pd.read_excel(file, sheet_name=7)
#%%
pop2 = np.array(data30)
pop1 = pop2.flatten()
pop30 = pop1.astype(float)
# sample_size = needs to be an exert from the data, or should it just be all the data?
#%% Requires the mean and standard deviation of the data
mean30 = np.mean(pop30)
stdev = np.std(pop30)
#%% Taking a subset of samples for running bootstrap on
    # Only necessary with very large datasets
# take 1k 'samples' from the larger pop
samples30 = pop30[:1000]
#%% Answer should be 100, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples30, stat_func=bs_stats.mean))
#%% Important Variables
bs_mean30 = bs.bootstrap(samples30, stat_func=bs_stats.mean)
#%% Important Null Variables
n30=[0.5 if x == 0 else x for x in samples30]
r301=np.prod(n30)
r302 = bs_mean30*r301
print(r302)
#%%
"""
Code for 40% Canopy Cover
"""
#%% Inserting the Dataset
# pop = Insert from Excel
file = "C:\\Users\Miles\Downloads\Project_Dataset.xlsx"
data40 = pd.read_excel(file, sheet_name=8)
#%%
pop2 = np.array(data40)
pop1 = pop2.flatten()
pop40 = pop1.astype(float)
# sample_size = needs to be an exert from the data, or should it just be all the data?
#%% Requires the mean and standard deviation of the data
mean40 = np.mean(pop40)
#%% Taking a subset of samples for running bootstrap on
    # Only necessary with very large datasets
# take 1k 'samples' from the larger pop
samples40 = pop40[:1000]
#%% Answer should be 100, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples40, stat_func=bs_stats.mean))
#%% Important Variables
bs_mean40 = bs.bootstrap(samples40, stat_func=bs_stats.mean)
#%% Important Null Variables
n40=[0.5 if x == 0 else x for x in samples40]
r401=np.prod(n40)
r402 = bs_mean40*r401
print(r402)
#%%
"""
Code for 50% Canopy Cover
"""
#%% Inserting the Dataset
# pop = Insert from Excel
file = "C:\\Users\Miles\Downloads\Project_Dataset.xlsx"
data50 = pd.read_excel(file, sheet_name=9)
#%%
pop2 = np.array(data50)
pop1 = pop2.flatten()
pop50 = pop1.astype(float)
# sample_size = needs to be an exert from the data, or should it just be all the data?
#%% Requires the mean and standard deviation of the data
mean = np.mean(pop50)
stdev = np.std(pop50)
#%% Taking a subset of samples for running bootstrap on
    # Only necessary with very large datasets
# take 1k 'samples' from the larger pop
samples50 = pop50[:1000]
#%% Answer should be 100, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples, stat_func=bs_stats.mean))
#%% Important Variables
bs_mean50 = bs.bootstrap(samples, stat_func=bs_stats.mean)
#%% Important Null Variables
n50=[0.5 if x == 0 else x for x in samples50]
r501=np.prod(n50)
r502 = bs_mean50*r501
print(r502)
#%%
"""
Code for 100% Canopy Cover
"""
#%% Inserting the Dataset
# pop = Insert from Excel
file = "C:\\Users\Miles\Downloads\Project_Dataset.xlsx"
data100 = pd.read_excel(file, sheet_name=10)
#%%
pop2 = np.array(data100)
pop1 = pop2.flatten()
pop100 = pop1.astype(float)
# sample_size = needs to be an exert from the data, or should it just be all the data?
#%% Requires the mean and standard deviation of the data
mean = np.mean(pop100)
stdev = np.std(pop100)
#%% Taking a subset of samples for running bootstrap on
    # Only necessary with very large datasets
# take 1k 'samples' from the larger pop
samples100 = pop100[:1000]
#%% Answer should be 100, fluctuates each time within a single range (provided)
print(bs.bootstrap(samples, stat_func=bs_stats.mean))
#%% Important Variables
bs_mean100 = bs.bootstrap(samples, stat_func=bs_stats.mean)
#%% Important Null Variables
n100=[0.5 if x == 0 else x for x in samples100]
r1001=np.prod(n100)
r1002 = bs_mean100*r1001
print(r1002)