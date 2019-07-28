# -*- coding: utf-8 -*-
"""
Created on Fri Oct 19 20:12:12 2018

@author: Ashlin
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
mydata=pd.read_csv('Data.csv')
print mydata.head()
print mydata.iloc[:,0]


france_vs_germany = {'France':0,'Spain':1,'Germany':2}
mydata['CountryCategory']=mydata.iloc[:,0].map(france_vs_germany)
print mydata.head()
X=mydata.iloc[:,0:3].values
y=mydata.iloc[:,3].values
from sklearn.preprocessing import Imputer
imputer =Imputer(missing_values='NaN', strategy='mean',axis=0)
imputer= imputer.fit(X[:,1:3])
X[:,1:3]=imputer.transform(X[:,1:3])
print X
mydata.info()
for col in list(mydata.columns):
    if('Purchased' in col):
        mydata[col]=mydata[col].astype(float)
mydata.info()
