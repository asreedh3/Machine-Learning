# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 18:23:16 2019

@author: Ashlin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import nltk

def confusionmatrix(Actual,Prediction):
    crosstab=pd.DataFrame({'Actual':Actual,'Prediction':Prediction})
    cmatrix=pd.crosstab(crosstab['Actual'],crosstab['Prediction'])
    print(cmatrix)
#getting a list of all the irrelavant /stop words
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer  #used for stemming


mydata=pd.read_csv("Restaurant_Reviews.tsv",delimiter='\t',quoting=3)
# changing the delimiter because it is a tab delimited file
#quoting parameter=3 ignores double quotes in the sentence structure
#normally double quotes are added if it is a csv to indicate that data belongs to particular column inspite of the comma limiter present in the sentence structure


# Clean the texts to create the bag of words model

import re # This library is specially used to aid in the cleaning of the text
corpus=[] #Corpus is a list of text in NLP

for i in range(0,1000):
    review=re.sub('[^a-zA-Z]',' ', mydata['Review'][i])
    #Only keeps the letters in the review, removes all the numbers and the punctuations 
    # We are specifying all the characters we dont want to remove
    # We dont want to remove all lower case and upper case letters from a to z
    # Here only working with the first review
    # Here when we remove a punctuation or a number the surrounding characters
    # are going to stick together and it will produce a word that does not make sense
    # To avoid that we need to remave the removed word with a space. 
    # Removed characters replaced by a space
    
    review=review.lower()
    #putting all the letters in the review in lower case
    
    review=review.split()
    #Splitting the review down into individual components
    ps=PorterStemmer()
    #Loading the Stemmer object
    review=[ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    #Performing Stemming on each word one word at a time
    #Only putting those words back in review that are not in the stop words list
    #We are only looking at english words to reduce computational complexity
    #removing all the articles, prepositions and connectors words, All non-essential words removed
    # You are trying to get rid of as many words as you can so as make the word matrix less sparse
    # Converting from list to set because python algorithms are faster searching through
    # a set rather than a list. Especially for longer reviews or text
    
    
    #Now converting a list of list back into a string
     
    review=' '.join(review) #tab seperated join to prevent the words all adding together to make one big string
    corpus.append(review)
    
# Creating the bag of words model
# You create the bag of words model from the Corpus. This done through Tokenization


from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
# The Count Vectorizer has a max_features parameter which can give you a way to reduce unique words that don't appear that frequently
# Just trying to reduce the columns
# Input is the number of features we want
#CountVectorizer token_pattern is basically the pattern that we want to keep

X=cv.fit_transform(corpus).toarray()
#to array function is done to get a numpy array

y=mydata['Liked']

from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import roc_curve,auc
from sklearn.metrics import matthews_corrcoef

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=100)

from sklearn.naive_bayes import GaussianNB

bayes_classifier=GaussianNB()
bayes_classifier.fit(X_train,y_train)
yhat=bayes_classifier.predict(X_test)
confusionmatrix(y_test,yhat)
print("The test accuracy for this classifier is: %0.3f" %bayes_classifier.score(X_test,y_test))
print("The precision score for this model is %0.3f" %(precision_score(y_test,yhat)))
print("The recall score for this model is %0.3f" %(recall_score(y_test,yhat)))
print("The f1 score for this model is %0.3f" %(f1_score(y_test,yhat)))
print("The matthews correlation score for this model is %0.3f" %(matthews_corrcoef(y_test,yhat)))
fpr, tpr, thresholds = roc_curve(y_test,yhat)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()
from sklearn.ensemble import RandomForestClassifier

random_forest=RandomForestClassifier()
from sklearn.model_selection import GridSearchCV

parameters={'n_estimators':[50,100,150,200],'max_depth':[10,12]}

rf_grid=GridSearchCV(random_forest,parameters,cv=5,n_jobs=2,verbose=2)
rf_grid.fit(X_train,y_train)
yhat=rf_grid.predict(X_test)
confusionmatrix(y_test,yhat)
print("The test accurcay score is: %0.3f" %(rf_grid.score(X_test,y_test)))
print("The best parameters are:%s" %(rf_grid.best_params_))

print("The test accuracy for this classifier is: %0.3f" %bayes_classifier.score(X_test,y_test))
print("The precision score for this model is %0.3f" %(precision_score(y_test,yhat)))
print("The recall score for this model is %0.3f" %(recall_score(y_test,yhat)))
print("The f1 score for this model is %0.3f" %(f1_score(y_test,yhat)))
print("The matthews correlation score for this model is %0.3f" %(matthews_corrcoef(y_test,yhat)))
rf_best=rf_grid.best_estimator_
fpr, tpr, thresholds = roc_curve(y_test,yhat)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, color='red', lw=1, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.show()