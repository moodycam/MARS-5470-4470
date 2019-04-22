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
sample_size = ran.choice(pop) # Gotta work on!
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
#%%



#%%
""" Creating and Adding Dataset """

"""
Set up binary capture/non-capture dataset
    Either a 0 or 1 for the day of monitoring
        Non-capture = 0
        Capture = 1
    Missed occurrances are not included
"""
#%% Inserting the Dataset
# pop = Insert from Excel
file = "C:\\Users\Miles\Downloads\Project_Dataset.xlsx"
data = pd.read_excel(file, sheet_name=4)
#%%


#%%
""" Coding Equations """

"""
1. Break into 2 groups
    Two different equations, one for sites that had at least one capture, (at
    least one 1) and another for sites with just 0's
2. Create functions that conduct equations
    First equation for groups with at least one capture and second equation
    for groups with just 0's
        Need to learn what the hell some of the symbols in these equations are
    p variabe at all sites will be constant
        0.5 is recommended by MacKenzie so will be used for now
        Will use temperature as an indicator for future analysis
            Was not measured for this dataset
3. Combine the 2 models
    Combine the two models so that they can be used against the vegetation
    map to create an occupancy model map
"""
#%% Important Variables
bootstrap_mean = 
#%% Important Null Variables

#%%
for i in data:
    