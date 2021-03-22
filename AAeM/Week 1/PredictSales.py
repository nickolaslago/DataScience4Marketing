# Simple Linear Regression example
# Problem: Predict sales based on different marketing platforms ads spending

# Import packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import datasets, linear_model
from sklearn.metrics import r2_score, mean_absolute_error

# Load dataset (from https://www.kaggle.com/ashydv/advertising-dataset)
ds = pd.DataFrame(pd.read_csv("advertising.csv"))
ds.head()

# Describe dataset
ds.describe()

# Data preparation
cols = ['TV','Radio','Newspaper']
X = ds[cols]
y = ds['Sales']

# Modeling
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size = 0.75, test_size = 0.25, random_state = 123)
regr = linear_model.LinearRegression() # create object
regr.fit(X_train, y_train) # train the model

# Evaluation
y_pred = regr.predict(X_test) # predict y for X_test

print('Coefficients: \n', regr.coef_)
print('Mean Absolute Error: %.2f' % mean_absolute_error(y_test, y_pred))
print('Coefficient of determination (r2): %.2f' % r2_score(y_test, y_pred))
