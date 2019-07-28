# -*- coding: utf-8 -*-
"""
Created on Thu Jan 24 20:36:34 2019

@author: Ashlin
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import keras
from sklearn.metrics import accuracy_score
mydata=pd.read_csv("Churn_Modelling.csv")
y=mydata.iloc[:,-1]
X=mydata.iloc[:,3:13]
X=pd.get_dummies(X)
scaler=StandardScaler()
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=0)
scaler.fit(X_train)
X_train=scaler.transform(X_train)
X_test=scaler.transform(X_test)


def confusionmatrix(Actual,Prediction):
    cmatrix=pd.DataFrame({'Actual':Actual,'Prediction':Prediction})
    crosstab=pd.crosstab(cmatrix['Actual'],cmatrix['Prediction'])
    return(crosstab)
    
# You need to import a sequential module that will initialize th neural network

from keras.models import Sequential

# You import this from keras to build the layers of the ANN

from keras.layers import Dense

# Initializing the ANN
# Two ways this can be done either by define the layers or defining the graph
# Here we initialize as a sequnce of layers

classifier=Sequential() # You specify the layers later

# Adding Input Layer and first Hidden Layer

classifier.add(Dense(6,input_dim=13,kernel_initializer='uniform',activation='relu'))
# 7 is the number of hidden nodes see why in comments below
#kernel_initializer is the what you use to initialize the starting weights, They are kept small and obtained using a uniform distribution
#input_dim should be number of independent variables
# You need to specify the input_dim only for first hidden layer
# WHen you create subsequent hidden layers you dont have to specify it anymore.
#Data is linearly separable, you don’t even need a hidden layer. 
# For the number of nodes in the first hidden layer you use the average of the number of input nodes and the number output nodes. Not a rule just a suggestion


# Adding another hidden layer

classifier.add(Dense(6,kernel_initializer='uniform',activation='relu'))


#Adding the output layer
# Only one output node here because binary class
#If you have more than one class output nodes= number of classes and activation=softmax
classifier.add(Dense(1,kernel_initializer='uniform',activation='sigmoid'))


# Compiling the ANN (Applying Stochastic Gradient Descent on the entire neural network)

classifier.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])
# optimizer: The algorithm you want to use to find the optimal weights in the ANN
# adam is a stochastic gradient descent algorithm 
# loss is the loss fucntion for the adam algorithm
# Since output activation is sigmoid we use logarithmic loss for loss function
# Binary outcomes = binary_crossentropy
# Multiple Classes = categorical_crossentropy
# metrics: criterion you choose to improve your models, expects a list of metrics hecne the square brackets
# When weights updated after each observation or a batch of observation you use this metric to evaluate



# Fitting the training set tot he ANN

classifier.fit(X_train,y_train,batch_size=10,epochs=100)

#batch_size: how many observations are passed through the ANN before weights are updated
#epochs: How many times the entire training set passes through the ANN

y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)

print(confusionmatrix(y_test,y_pred[:,0]))
print(accuracy_score(y_test,y_pred[:,0]))