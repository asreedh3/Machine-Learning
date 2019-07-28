# -*- coding: utf-8 -*-
"""
Created on Thu Dec 20 21:43:48 2018

@author: Ashlin
"""

import pandas as pd
import numpy as np
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
mydata=pd.read_csv("Position_Salaries.csv")
y=mydata.iloc[:,2:3]
X=mydata.iloc[:,1:2]
# We scale both the X and y values because this is a regression problem. 
# For a classificatiopn problem only X needs to be scaled

scaler_X=StandardScaler()
scaler_y=StandardScaler()
scaler_X.fit(X)
X=scaler_X.transform(X)
y=scaler_y.fit_transform(y)
y=y.reshape(len(y))
svr=SVR(kernel='rbf')
svr.fit(X,y)

yhat=svr.predict(X)

plt.title("Regression Plot")
plt.xlabel("X")
plt.ylabel("Y")
plt.scatter(X,y,color='blue',label='Actual')
plt.plot(X,yhat,color='red',label='SVR')
plt.legend()
plt.show()

predict=scaler_X.transform(6.5)
prediction=svr.predict(predict)
prediction=scaler_y.inverse_transform(prediction)

plt.title("Regression Plot")
plt.xlabel("X")
plt.ylabel("Y")
X_grid=np.arange(min(X),max(X),0.1)
X_grid=X_grid.reshape(len(X_grid),1)
plt.scatter(X,y,color='blue',label='Actual')
plt.plot(X_grid,svr.predict(X_grid),color='red',label='SVR')
plt.legend()
plt.show()


