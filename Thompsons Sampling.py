# -*- coding: utf-8 -*-
"""
Created on Thu Jan 17 17:05:01 2019

@author: Ashlin
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import math as mt
import random
mydata=pd.read_csv("Ads_CTR_Optimisation.csv")

ad_selected=[]
N=10000
total_reward=0
d=10
number_of_rewards0=[0]*d
number_of_rewards1=[0]*d
for n in range(0,N):
    max_random_success_prob=0
    ad=0
    for i in range(0,d):
        random_beta=random.betavariate(number_of_rewards1[i]+1,number_of_rewards0[i]+1)
        
        if random_beta>max_random_success_prob:
            max_random_success_prob=random_beta
            ad=i
    total_reward=mydata.values[n,ad]+total_reward
    ad_selected.append(ad)
    if mydata.values[n,ad]==0:
        number_of_rewards0[ad]=number_of_rewards0[ad]+1
    else:
        number_of_rewards1[ad]=number_of_rewards1[ad]+1

plt.hist(ad_selected,edgecolor='black')
plt.title("Histogram")
plt.xlabel("Ad Selected")
plt.ylabel("Number of Times Ad Selected")
plt.show()
    
            
    
    
    
    
