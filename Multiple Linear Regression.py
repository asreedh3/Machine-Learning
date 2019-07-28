# -*- coding: utf-8 -*-
"""
Created on Sat Dec 15 14:56:23 2018

@author: Ashlin
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
mydata=pd.read_csv("50_Startups.csv")
y=mydata.iloc[:,-1]
X=mydata.iloc[:,0:4]
X=pd.get_dummies(X)

# Dropping a dummy variable to prevent multicollinearity assumption violation for regression

X=X.drop(columns=['State_New York'])
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
modelnow=LinearRegression()
modelnow.fit(X_train,y_train)
yhat=modelnow.predict(X_test)
r2score=r2_score(y_test,yhat)
print('The R squared for the model is %0.5f' % (r2score))

#Backward Elimination Regression

import statsmodels.formula.api as sm

#need to add the column for the bo constant which is just a column of 1s

#In the Linear Regression class the column of 1s is put by the class dont have to enter it manuallyh

one_matrix=np.ones((50,1))
one_matrix=pd.DataFrame(one_matrix)

#make a matrix of 1s and add them together use concat ignore_index and reseti, axis=1 along column

X=pd.concat([one_matrix,X],ignore_index=True,axis=1)

#Creating an Intial X matrix of all the features and then removing them Backwards

X_opt=X.iloc[:,[0,1,2,3,4,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary() #gives anova based summary
 

X_opt=X.iloc[:,[0,1,2,3,5]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary() #gives anova based summary

X_opt=X.iloc[:,[0,1,2,3]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary() #gives anova based summary

X_opt=X.iloc[:,[0,1,3]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary() #gives anova based summary

X_opt=X.iloc[:,[0,1]]
regressor_ols=sm.OLS(endog=y,exog=X_opt).fit()
regressor_ols.summary() #gives anova based summary
 
def backward_elimination(X_new,sl):
    variable_count=len(X.iloc[0,:])
    for i in range(0,variable_count):
        regressor_ols=sm.OLS(endog=y,exog=X).fit()
        maxvar=max(regressor_ols.pvalues)
        if maxvar>sl:
            for j in range(1,variable_count+1):
                if (regressor_ols.pvalues[j]==maxvar):
                    X_new=X_new.drop(X_new.columns[j],axis=1)
    regressor_ols.summary()
    return(X_new)

sl=0.05
X_modelled=backward_elimination(X_opt,sl)