# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:53:12 2019

@author: Miles
"""
#%%
"""
Correlation is X, Y of two datasets
    test of linear relationship
    assumes normal distribution of data
    finds covariance and divides that by the standard deviations of x and y
        covariance = sums how each datapoint differs from the x and y means
        divides the sums by the standard deviations of x and y
    gives you a decreasing, 0, or increasing line
        r = 1 it is a positive relationship
        r = -1 it is a negative relationship
        r = 0 there is no relationship
    can do X or Y first, i.e. rxy = ryx
    want to find two things:
        1) a simple model based on the correlations
            y = mx + b
                want to find coefficents in our model: m and b (b is the intercept)
                find m and b by calculating the residules and minimizing them
                    residules are the difference between the model and the data (lines from the curve)
                    residules are usually squared (makes sure positive and negatives residules don't cancel each other out, and let's you know how much things cost to be included in the model)
                    variance of the data (sum of squares) (SSD) = sum of (data model - the model value)^2
                    residual sum of squares (SSR) = sum of (data model - ymi)^2
        2) how good is my model?
            how well does it fit the data?
            R^2 = 1 - (SSR / SSD)
"""
#%%
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#%% just uploads dataset
file = "C:\\Users\Miles\Downloads\Must_Sort\movies.xls"
movies_sheet1 = pd.read_excel(file, sheet_name=0, index_col=0)
movies_sheet2 = pd.read_excel(file, sheet_name=1, index_col=0)
movies_sheet3 = pd.read_excel(file, sheet_name=2, index_col=0)
movies = pd.concat([movies_sheet1, movies_sheet2, movies_sheet3])
#%% Plot budgets vs. Gross Earnings
plt.scatter(movies['Budget'], movies['Gross Earnings'])
plt.plot([0,1E10], [0, 1E10], 'k--') # Plots the y=x line which is SUPER helpful
plt.xlabel('Budget')
plt.ylabel('Gross Earnings')
plt.xlim([0,0.1E10])
plt.ylim([0,1E9])
#%% Finds the correlations between all the variables
# Pandas allow us to get the correlations of the entire dataframe all at once. Numpy will also do this.
corr = movies.corr() # to see it, save it as a variable and open it using variable explorer
#%%
""" Linear Regression """
#%%
from sklearn import datasets # imports datasets from scikit-learn
data = datasets.load_boston()
#%%
print(data.DESCR)
#%% Convert the data set into a pandas data frame
df = pd.DataFrame(data.data, columns=data.feature_names)
#%% Put the target (housing value ==MEDV) in another DataFrame
target = pd.DataFrame(data.target, columns=["MEDV"])
#%% Are room numbers and value correlated?
np.corrcoef(df["RM"], target["MEDV"])
#%%
plt.scatter(df["RM"], target["MEDV"])
plt.xlabel('Number of Rooms')
plt.ylabel('House value ($1,000s)')
#%%
""" Doing a linear regressions using scipy """
#%%
from scipy import stats
#%% Creates multiple variables and we provide names for those variables, which saves them
slope, intercept, r_value, p_value, std_err = stats.linregress(df["RM"], target["MEDV"])
#%%
slope
#%%
intercept
#%%
r_value
#%%
p_value
#%%
std_err
#%%
plt.plot(df["RM"], slope*df["RM"]+intercept, 'k--')
plt.scatter(df["RM"], target["MEDV"])
plt.xlabel('Number of Rooms')
plt.ylabel('House value($1000s)')
#%%
import statsmodels.api as sm
#%%
X = df["RM"] # what we think the cost depends on
y = target["MEDV"]
#%% Note: the target goes first
model = sm.OLS(y, X).fit()
#%% this fits y = aX with no constant (0 intercept)
model.summary()
#%% make the predictions by the model
predictions = model.predict(X)
#%% model with NOT a great fit
plt.plot(X, predictions, 'k--')
plt.scatter(df["RM"], target["MEDV"])
plt.xlabel('Number of Rooms')
plt.ylabel('House Value ($1,000s)')
#%%
"""
Exercise 3:
    do the tutorial:
    https://towardsdatascience.com/linear-regression-on-boston-housing-dataset-f409b7e4a155
"""
#%% Import required libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
#%% Loads the housing data from the scikit-learn library
from sklearn.datasets import load_boston
boston_dataset = load_boston()
#%% Print the value of the dataset to understand what it contains
print(boston_dataset.keys())
#%% MEDV is the feature variable we are interested in
boston_dataset.DESCR
#%%
"""
Note:
    Can also click on 'boston_dataset' in the variable explorer and then click
    on 'feature names' to gain a similar list that is much easier to read.
"""
#%% will load the data into Pandas using a dataframe
boston = pd.DataFrame(boston_dataset.data, columns=boston_dataset.feature_names)
boston.head()
#%% 'MEDV' is missing from the data. Create a new column of target values and add it to the dataframe
boston['MEDV'] = boston_dataset.target
#%% Find out if there are any null variables
boston.isnull().sum()
#%% Exploratory data analysis
sns.set(rc={'figure.figsize': (11.7, 8.27)})
sns.distplot(boston['MEDV'], bins=30)
plt.show()
# Histogram shows MEDV is distributed normally with a few outliers, mainly at the high end
#%% Create a correlation matrix that measures the linear relationships between the variables.
# Note: uses the heatmap function from the seaborn library to plot the correlation matrix.
correlation_matrix = boston.corr().round(2)
# annot=True to print the values inside the square
sns.heatmap(data=correlation_matrix, annot=True)
#%%
"""
Note:
    To fit a linear regression model we select those features which have a high
    correlation with out target variable (MEDV).
    Corretlation matrix tells us RM has a strong positive correlation, whereas
    LSTAT has an extreme negative.
    Make sure not to use two variables that are very similar, for example, RAD
    and TAX. Only 1 is needed.
"""
#%%
plt.figure(figsize=(20,5))

features = ['LSTAT', 'RM']
target = boston['MEDV']

for i, col in enumerate(features):
    plt.subplot(1, len(features), i+1)
    x = boston[col]
    y = target
    plt.scatter(x,y, marker='o')
    plt.title(col)
    plt.xlabel(col)
    plt.ylabel('MEDV')
#%%
"""
Note:
    Prices increase as the value of RM increases linearly. There are few
    outliers and the data seems to be capped at 50.
    Prices tend to decrease with an increase in LSTAT. Though it doesn't
    look to be following exactly a linear model (curves).
"""
#%% Concatenate the LSTAT and RM columns using np.c_ provided by the numpy library
X = pd.DataFrame(np.c_[boston['LSTAT'], boston['RM']], columns=['LSTAT', 'RM'])
Y = boston['MEDV']
#%% Splitting the datat into training and testing sets
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X,Y, test_size = 0.2, random_state=5)
print(X_train.shape)
print(X_test.shape)
print(Y_train.shape)
print(Y_test.shape)
#%% Training and testing the model
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

lin_model = LinearRegression()
lin_model.fit(X_train, Y_train)
#%% Model Evaluation
from sklearn.metrics import r2_score # they forgot to throw that little bit in
# model evaluation for training set
y_train_predict = lin_model.predict(X_train)
rmse = (np.sqrt(mean_squared_error(Y_train, y_train_predict)))
r2 = r2_score(Y_train, y_train_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")


#model evaluation for testing set
y_test_predict = lin_model.predict(X_test)
rmse = (np.sqrt(mean_squared_error(Y_test, y_test_predict)))
r2 = r2_score(Y_test, y_test_predict)

print("The model performance for training set")
print("--------------------------------------")
print('RMSE is {}'.format(rmse))
print('R2 score is {}'.format(r2))
print("\n")
#%%
"""
--- Scrum ---
  :Presented by Cameron
Project Management:
  > 4 major principles
    1.) Work with what you have
        Don't waste time, figure out what can be done now
        Daily check-ins
    2.) Constant Feedback
        Weekly reviews and evaluations
        What you finished, didn't finish, and how to change that
    3.) Workable goals
        Overall project may have lofty goals, but have phases
        Every day should have goals to work toward
        Clearly defined and make sense for you
        Goals you can work toward every day
    4.) SPRINT!
        In rugby its where you run with the ball
        When you've gone through all of the steps from 1 to 3 you have to
        actually do the damn thing.
        Specific time cut out of the day.
  > Values of Scrum (Apparently they have a website? [scrum.org])
    1.) Courage
    2.) Focus
    3.) Commitment
    4.) Respect
    5.) Openness
"""
