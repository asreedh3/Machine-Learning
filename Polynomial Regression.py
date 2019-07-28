# -*- coding: utf-8 -*-
"""
Created on Wed Dec 19 20:56:04 2018

@author: Ashlin
"""

import pandas as pd
import numpy as np
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
mydata=pd.read_csv('Position_Salaries.csv')
y=mydata.iloc[:,-1]

# This is done so that we can make X as a matrix and y as a vector. Done to prevent warnings of any kind
X=mydata.iloc[:,1:2]
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
lregressor=LinearRegression()
lregressor.fit(X,y)
print (r2_score(y,lregressor.predict(X)))

plt.title("Linear Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X,y)
plt.plot(X,lregressor.predict(X))
plt.show()

# Polynomial Regression. Done by adding polynomial features

from sklearn.preprocessing import PolynomialFeatures

# Polynomial Features Class just gives you poly features of a user specified degree
poly_features=PolynomialFeatures(4)

# You Transform X to have poly features along with interactions
X_poly=poly_features.fit_transform(X)
pregressor=LinearRegression()
pregressor.fit(X_poly,y)
print(r2_score(y,pregressor.predict(X_poly)))

plt.title("Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X,y)
plt.plot(X,pregressor.predict(X_poly))
plt.show()


plt.title("Linear Regression Vs Polynomial Regression")
plt.xlabel("X")
plt.ylabel("Y")
X_grid=np.arange(min(X.values),max(X.values),0.1)
# Need to reshape the vector and make it a matrix. X as a vector has issues
X_grid=X_grid.reshape(len(X_grid),1)
'''plt.scatter(X_grid,y,color='blue',label="Actual Salary")'''
plt.plot(X_grid,lregressor.predict(X_grid),color='red',label="Linear")
plt.plot(X_grid,pregressor.predict(poly_features.fit_transform(X_grid)),color='green',label="Polynomial")
plt.legend()
plt.show()


print (lregressor.predict(6.5))

print (pregressor.predict(poly_features.fit_transform(6.5)))
