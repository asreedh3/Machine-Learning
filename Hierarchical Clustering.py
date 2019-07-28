# -*- coding: utf-8 -*-
"""
Created on Sun Jan  6 18:17:17 2019

@author: Ashlin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from matplotlib.colors import ListedColormap

mydata=pd.read_csv("Mall_Customers.csv")
X=mydata.iloc[:,3:5].values
dendogram=sch.dendrogram(sch.linkage(X,method='ward',metric='euclidean'))
plt.title("Dendogram")
plt.xlabel("Customers or Points of Data")
plt.ylabel("Euclidean Distance")
plt.show()

from sklearn.cluster import AgglomerativeClustering
hc=AgglomerativeClustering(n_clusters=5,affinity='euclidean',linkage='ward')
yhat=hc.fit_predict(X)

for i in range(0,5):
    plt.scatter(X[yhat==i,0],X[yhat==i,1],s=100,c=ListedColormap(('red','green','blue','cyan','magenta'))(i),label="Cluster Number %d" %(i))

plt.title("Hierarchical Clustering")
plt.xlabel("Annual Income")
plt.ylabel("Spending Score")
plt.legend()
plt.show()