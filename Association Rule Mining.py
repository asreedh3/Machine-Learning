# -*- coding: utf-8 -*-
"""
Created on Fri Jan 11 22:56:26 2019

@author: Ashlin
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
mydata=pd.read_csv("Market_Basket_Optimisation.csv",header=None)
#creating a list of lists
transactions=[]
for i in range(0,7501): #upper bound excluded
    transactions.append([str(mydata.values[i,j]) for j in range(0,20)])
    # apriori fucntion expecting a string
    
# Importing custom association rule mining functionality
# transactions need to be a list of a list of items
from apyori import apriori
# Need to set the argumentbased on your business problems 
rules=apriori(transactions,min_support=0.003,min_confidence=0.2,min_lift=3,min_length=2)
#rules obtained are already sorted on a combination of criteria like support, confidence and lift
results=pd.DataFrame(rules)

rule_list=[]
for i in range(0,len(results)):
    rule_list.append('RULE:'+str(results.iloc[i,2][0][0])+','+str(results.iloc[i,2][0][1])+'\nSUPPORT:'+str(results.iloc[i,1])+'\nCONFIDENCE:'+str(results.iloc[i,2][0][2])+'\nLIFT:'+str(results.iloc[i,2][0][3]))
# the + opeartor is just to add 2 strings together and the \n is new line opeartor