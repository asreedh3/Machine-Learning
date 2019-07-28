# -*- coding: utf-8 -*-
"""
Created on Sun Dec  9 21:51:47 2018

@author: Ashlin
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt
mydata=pd.read_csv('Salary_Data.csv')
X=pd.DataFrame(mydata['YearsExperience'])
y=pd.DataFrame(mydata['Salary'])
seed=5000
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=1/3,random_state=seed)
modelnow=LinearRegression()
modelnow.fit(X_train,y_train)
modelnow.coef_
modelnow.intercept_
modelnow.score(X_train,y_train)
yhat=modelnow.predict(X_test)
print (r2_score(y_test,yhat))
plt.scatter(X_train,y_train,color='red')
plt.plot(X_train,modelnow.predict(X_train),color='blue')
plt.title('Salary vs Experience (Training Set)')
plt.xlabel('Years of Experience')
plt.ylabel("Salary")
plt.show() #end of the graph and ready to plot it

plt.scatter(X_test,y_test, color='red')
plt.plot(X_train,modelnow.predict(X_train), color='blue') #equation of the regression line remains the same
plt.title('Salary vs Years of Experience (Test Set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()