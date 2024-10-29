#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ML1 In-Class
.py file
"""

# import packages
from pydataset import data
import pandas as pd
from sklearn.model_selection import train_test_split

#%%
mtcars = data('mtcars')
mtcars.head()

#%%
iris = data('iris')
iris.head()

#%%
# What mental models can we see from these data sets?
# What data science questions can we ask?

#%%
"""
Example: k-Nearest Neighbors
"""
# We want to split the data into train and test data sets. To do this, we will use sklearn's train_test_split method.
# First, we need to separate variables into independent and dependent dataframes.

X = iris.drop(['Species'], axis=1).values   # dependent variables
y = iris['Species'].values                  # independent variable

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
# we can change the proportion of the test size; we'll go with 1/3 for no

#%%
# Now, we use the scikitlearn k-NN classifier
from sklearn.neighbors import KNeighborsClassifier
neigh = KNeighborsClassifier(n_neighbors=3)
neigh.fit(X_train, y_train)

#%%
# now, we check the model's accuracy:
neigh.score(X_train, y_train)

#%%
# now, we test the accuracy on our testing data.
neigh.score(X_test, y_test)

#%%
"""
Patterns in data
"""
# Look at the following tables: do you see any patterns? How could a classification model point these out?
patterns = iris.groupby(['Species'])
patterns['Sepal.Length'].describe()

#%%
patterns['Sepal.Width'].describe()

#%%
patterns['Petal.Length'].describe()

#%%
patterns['Petal.Width'].describe()

#%%
"""
Mild disclaimer
"""
# Do not worry about understanding the machine learning in this example! We go over kNN models at length later in the course; 
# you do not need to understand exactly what the model is doing quite yet. For now, ask yourself:

# 1. What is the purpose of data splitting?
# 2. What can we learn from data testing/validation?
# 3. How do we know if a model is working?
# 4. How could we find the model error?

# If you want, try changing the size of the test data or the number of n_neighbors and see what changes!