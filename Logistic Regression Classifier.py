# -*- coding: utf-8 -*-
"""
Created on Fri Dec 28 17:26:28 2018

@author: Ashlin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix

def confusionmatrix(Actual,Prediction):
    cmatrix=pd.DataFrame({'Actual':Actual,'Predicted':Prediction})
    crosstab=pd.crosstab(cmatrix["Actual"],cmatrix["Predicted"])
    return(crosstab)
    
mydata=pd.read_csv("Social_Network_Ads.csv")
X=mydata.iloc[:,2:4]
y=mydata.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression(random_state=0)
classifier.fit(X_train,y_train)
yhat=classifier.predict(X_test)
print("The accuracy score for the given Classifier is: %0.3f" %(accuracy_score(y_test,yhat)))
print("The test score is: %0.3f" % (classifier.score(X_test,y_test)))
print ("The train score is: %0.3f" % (classifier.score(X_train,y_train)) )

confusion_matrix(y_test,yhat)
print(confusionmatrix(y_test,yhat))


from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import numpy as np
X_set,y_set=X_train,y_train

# We can only set up the mesh grid for 2-D, where we plot all the possible values for 
#attribute 1 and attribute 2

#Splitting the attributes up to designate classifier boundaries in terms of attribute 1 and 2
X1,X2=np.meshgrid(np.arange(min(X_set[:,0])-1,max(X_set[:,0])+1,0.01),np.arange(min(X_set[:,1])-1,max(X_set[:,1])+1,0.01))

# if we just use np array create two rows first X1 and second row X2.Need to transpose it hence
z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
# Need to reshape the array since all of them need to have same shape
z=np.reshape(z,X1.shape)
plt.contourf(X1,X2,z,alpha=0.75,cmap=ListedColormap(('red','green')))

# Setting the plot limits for the graph
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

# for an enumerate loop i is the counter and j is the iterable
# here j sepeartes X1 values that have y_set 0 and 1
# and it then does the same thing for X2
#Making sure the same row is selected in across X1,X2,y_set

'''What this y_set==1 is doing is gathering all the indexes corresponding to the y_set
where y_set=1 or 0. j takes two values 0 or 1. So sepeartes the 0 and 1 observations,the 
X_set[y_set==j,1] takes all the observations of the X2 feature sepearting them based on
y_set value
(i) is used to give a float value that corresponds to the ListedColormap
where i takes 0 for all the 0 values and hence gives red for the color list
and i takes 1 for all the 1 values and hence gives green from the color list'''

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title("Logistic Regression Training Set")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

X_set,y_set=X_test,y_test

# We can only set up the mesh grid for 2-D, where we plot all the possible values for 
#attribute 1 and attribute 2

#Splitting the attributes up to designate classifier boundaries in terms of attribute 1 and 2
X1,X2=np.meshgrid(np.arange(min(X_set[:,0])-1,max(X_set[:,0])+1,0.01),np.arange(min(X_set[:,1])-1,max(X_set[:,1])+1,0.01))

# if we just use np array create two rows first X1 and second row X2.Need to transpose it hence
z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
# Need to reshape the array since all of them need to have same shape
z=np.reshape(z,X1.shape)
plt.contourf(X1,X2,z,alpha=0.75,cmap=ListedColormap(('red','green')))

# Setting the plot limits for the graph
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())

# for an enumerate loop i is the counter and j is the iterable
# here j sepeartes X1 values that have y_set 0 and 1
# and it then does the same thing for X2
#Making sure the same row is selected in across X1,X2,y_set

for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title("Logistic Regression Training Set")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

