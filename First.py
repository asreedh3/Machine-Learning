# -*- coding: utf-8 -*-
"""
Created on Tue Oct 16 13:09:04 2018

@author: Ashlin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
mydata=pd.read_csv('Data.csv')
print mydata.head()
print mydata.iloc[:,0]

print mydata.head()
X=mydata.iloc[:,0:3].values
y=mydata.iloc[:,3].values
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values='NaN', strategy='mean',axis=0)
X[:,1:3]= imputer.fit_transform(X[:,1:3])

print X


from sklearn.preprocessing import LabelEncoder,OneHotEncoder
labelencoder=LabelEncoder()
X[:,0]=labelencoder.fit_transform(X[:,0])
onehotencoder=OneHotEncoder(categorical_features=[0])
X=onehotencoder.fit_transform(X).toarray()
print X
y=labelencoder.fit_transform(y)
print y
print X[:,2]
from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
X_train=scaler.fit_transform(X_train)
X_test=scaler.transform(X_test)
mydata.info()

