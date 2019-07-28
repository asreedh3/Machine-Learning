# -*- coding: utf-8 -*-
"""
Created on Sun Dec 30 15:47:00 2018

@author: Ashlin
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

def confusionmatrix(Actual,Predicted):
    cmatrix=pd.DataFrame({'Actual':Actual,'Prediction':Predicted})
    crosstab=pd.crosstab(cmatrix['Actual'],cmatrix['Prediction'])
    return(crosstab)

mydata=pd.read_csv("Social_network_Ads.csv")
X=mydata.iloc[:,2:4]
y=mydata.iloc[:,-1]
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.25,random_state=0)
scaler=StandardScaler()
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)

classifier=SVC(kernel='linear',random_state=0)
classifier.fit(X_train,y_train)
yhat=classifier.predict(X_test)
print("The training accuracy %0.3f" % (classifier.score(X_train,y_train)))
print("The test accuracy %0.3f" % (classifier.score(X_test,y_test)))

print(confusionmatrix(yhat,y_test))

X_set,y_set=X_train,y_train

X1,X2=np.meshgrid(np.arange(X_set[:,0].min()-1,X_set[:,0].max()+1,0.01),np.arange(X_set[:,1].min()-1,X_set[:,1].max(),0.01))

z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
z=np.reshape(z,X1.shape)
plt.contourf(X1,X2,z,alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.title("SVM Training Set Boundary")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

X_set,y_set=X_test,y_test

X1,X2=np.meshgrid(np.arange(X_set[:,0].min()-1,X_set[:,0].max()+1,0.01),np.arange(X_set[:,1].min()-1,X_set[:,1].max(),0.01))

z=classifier.predict(np.array([X1.ravel(),X2.ravel()]).T)
z=np.reshape(z,X1.shape)
plt.contourf(X1,X2,z,alpha=0.75,cmap=ListedColormap(('red','green')))
plt.xlim(X1.min(),X1.max())
plt.ylim(X2.min(),X2.max())
for i,j in enumerate(np.unique(y_set)):
    plt.scatter(X_set[y_set==j,0],X_set[y_set==j,1],c=ListedColormap(('red','green'))(i),label=j)

plt.title("SVM Test Set Boundary")
plt.xlabel("Age")
plt.ylabel("Estimated Salary")
plt.legend()
plt.show()

