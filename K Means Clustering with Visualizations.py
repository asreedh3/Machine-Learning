# -*- coding: utf-8 -*-
"""
Created on Fri Jan  4 17:18:57 2019

@author: Ashlin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.colors import ListedColormap
mydata=pd.read_csv("Mall_Customers.csv")
X=mydata.iloc[:,3:5].values

from sklearn.cluster import KMeans

wcss=[]

for i in range(1,11):
    kmeans=KMeans(n_clusters=i,init='k-means++',max_iter=300,n_init=10,random_state=0) 
    # k-means++ initialization to avoid bad random start points
    #max_iter: the number of iterations the Kmeans alogo is run for
    #n_int: the number of times Kmeans run with different initial centroids
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)

plt.plot(range(1,11),wcss)
plt.title("The Elbow method")
plt.xlabel("Number of Clusters")
plt.ylabel("WCSS")
plt.show()


kmeans=KMeans(n_clusters=5,init='k-means++',n_init=10,max_iter=300,random_state=0)
y_kmeans=kmeans.fit_predict(X)

# Visualizing the Clusters

for i in range(0,5):
    plt.scatter(X[y_kmeans==i,0],X[y_kmeans==i,1],c=ListedColormap(('red','green','blue','cyan','magenta'))(i),label='Cluster Number %d' %(i),s=100)
plt.scatter(kmeans.cluster_centers_[:,0],kmeans.cluster_centers_[:,1],s=300,c='yellow',label='Centroids')
plt.title("Clusters of Clients")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()
