# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 14:43:58 2019

@author: Ashlin
"""

from sklearn.decomposition import PCA
pca= PCA(n_components=3)
pca.fit(X)
pca.explained_variance_ratio_
pca.explained_variance_