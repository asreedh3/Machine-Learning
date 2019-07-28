# -*- coding: utf-8 -*-
"""
Created on Fri Dec 21 17:29:04 2018

@author: Ashlin
"""

import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import numpy as np
mydata=pd.read_csv("Position_Salaries.csv")
X=mydata.iloc[:,1:2]
y=mydata.iloc[:,-1]

regressor=RandomForestRegressor(max_features='sqrt',n_estimators=300,criterion='mse',random_state=0)
regressor.fit(X,y)

plt.title("Regression")
plt.xlabel("X")
plt.ylabel("Predicted Value")
X_grid=np.arange(min(X.values),max(X.values),0.01)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='blue',label="Actual")
plt.plot(X_grid,regressor.predict(X_grid),color='red',label="RFR")
plt.legend()
plt.show()

prediction=regressor.predict(6.5)
print("The predicted value for the Salary is %0.4f" % (prediction))