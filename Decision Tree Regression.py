# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 16:42:15 2018

@author: Ashlin
"""

import pandas as pd
from sklearn.tree import DecisionTreeRegressor
import matplotlib.pyplot as plt
import numpy as np
mydata=pd.read_csv('Position_Salaries.csv')
X=mydata.iloc[:,1:2]
y=mydata.iloc[:,-1]

regressor=DecisionTreeRegressor(random_state=0)
regressor.fit(X,y)

plt.title("Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X,y,color='blue',label="Actual")
plt.plot(X,regressor.predict(X),color='red')
plt.legend()
plt.show()


# SMoother plot for Regression

X_grid=np.arange(min(X.values),max(X.values),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='blue',label="Actual")
plt.plot(X_grid,regressor.predict(X_grid),color='red')
plt.title("Regression")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

prediction=regressor.predict(6.5)

print("The prediction obtained is %0.3f" % (prediction))