# -*- coding: utf-8 -*-
"""
Created on Sat Dec 29 21:33:46 2018

@author: Ashlin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
def confusionmatrix(Actual,Predicted):
    cmatrix=pd.DataFrame({'Actual':Actual,'Prediction':Predicted})
    crosstab=pd.crosstab(cmatrix['Actual'],cmatrix['Prediction'])
    return(crosstab)
mydata=pd.read_csv("Social_Network_Ads.csv")

X=mydata.iloc[:,2:4]
y=mydata.iloc[:,-1]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)

scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

classifier=KNeighborsClassifier(n_neighbors=5,p=2,metric='minkowski')
classifier.fit(X_train,y_train)
yhat=classifier.predict(X_test)
print (confusionmatrix(y_test,yhat))

print("The current training accuracy is %0.3f" % (classifier.score(X_train,y_train)))
print("The current test accuracy is %0.3f" % (classifier.score(X_test,y_test)))

X_set,y_set=X_train,y_train

X1,X2=np.meshgrid(np.arange(X_set[:,0].min()-1,X_set[:,0].max()+1,0.01),np.arange(X_set[:,1].min()-1,X_set[:,1].max()+1,0.01))

z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)

z=np.reshape(z,X1.shape)

plt.contourf(X1,X2,z,alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())


for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)
plt.title("K-NN Training Set")
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
plt.title("K-NN Test Set")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

