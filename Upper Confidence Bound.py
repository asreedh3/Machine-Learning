# -*- coding: utf-8 -*-
"""
Created on Tue Jan 15 17:13:10 2019

@author: Ashlin
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math as mt

# Trying to see which ad gives a better return using the Upper Confidence Bound algorithm for the multi armed bandit problem
mydata=pd.read_csv("Ads_CTR_Optimisation.csv")

# The dataset gives the actual outcomes for the user click data. These outcomes are something that we normally do not know

#implementing random selection for the ads

import random
N=10000
d=10
total_reward=0
ad_selected=[]
for i in range(0,N):
    ad=random.randrange(d)
    ad_selected.append(ad)
    total_reward=total_reward+mydata.values[i,ad]

plt.hist(ad_selected,edgecolor='black')
plt.title("Number of times the ad is selected")
plt.xlabel("The ad selected")
plt.ylabel("The times the ad is selected")
plt.show()

# Implementing UCB

number_of_selections= [0]*d
sum_of_rewards=[0]*d
ads_selected=[]
total_reward=0


# Algorith techniques for each round we needed to calculate certain parameters so
# For each rounds these parameters needed to be calcvulated for eahc of the 10 ads
#We dont need to store avergae reward it is just used for compuation to see which ad has the highest
for n in range(0,N): #rounds
    max_bound=0
    max_ad=0
    
    for i in range(0,d):#ads computing average reward and confidence interval
        if number_of_selections[i]>0:
         average_reward= sum_of_rewards[i]/number_of_selections[i]
         confidence_interval=mt.sqrt((3*(mt.log(n+1)))/(2*number_of_selections[i]))
         upper_bound=average_reward+confidence_interval
        else:
            upper_bound=1e400
        
            
        if upper_bound>max_bound:
            max_ad=i
            max_bound=upper_bound
    
    ads_selected.append(max_ad)
    #list best to not use a contraction method of addition
    number_of_selections[max_ad]=number_of_selections[max_ad]+1
    sum_of_rewards[max_ad]=sum_of_rewards[max_ad]+ mydata.values[n,max_ad]
    total_reward=total_reward+mydata.values[n,max_ad]


plt.hist(ads_selected,edgecolor='black')

plt.xlabel("Ads")


plt.ylabel("Number of Times Ad Selected")
plt.title("Histogram")
plt.show()

    