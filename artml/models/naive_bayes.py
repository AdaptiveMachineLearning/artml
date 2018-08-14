
# coding: utf-8

# In[1]:

# Importing all the required libraries
import os
import math
from numpy import * 
import numpy as np
import pandas as pd
from sklearn import datasets
from scipy import stats
from scipy.stats import norm
from scipy.stats import chisqprob
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[4]:

class bayes_numerical():
    
    def fit(self, BET, X ,target):

        l =(len(BET))
        BET.reset_index(drop = True, inplace = True)
        x = BET.to_dict(orient='list')
        keys =list(x.keys())

        probability = []
        likelihood = 1
        att_prior_prob = 1
        class_prior_prob = 1
        for i in range(len(BET)):
            if keys[i] != target:
                count = x[target][i][6]
                sumxy = x[target][i][10]
                sumxy2 = x[target][i][11]
                Mean = sumxy/count
                Variance = (sumxy2 - (((sumxy)**2)/count))/count
                value = X[i]
                likelihood = likelihood*(1/math.sqrt(2*np.pi*Variance))*(np.e**(-(value-Mean)/(2*Variance)))

                class_prior_prob = (count/x[target][i][5])

                count_att = x[target][i][0]
                sumxy_att = x[target][i][1]
                sumxy2_att = x[target][i][2]
                Mean_att = sumxy_att/count_att
                Variance_att = (sumxy2_att - (((sumxy_att)**2)/count_att))/count_att

                att_prior_prob = att_prior_prob*(1/math.sqrt(2*np.pi*Variance_att))*(np.e**(-(value-Mean_att)/(2*Variance_att)))

        self.post_prob = (class_prior_prob * likelihood)/att_prior_prob

        return self.post_prob


# In[5]:

class bayes_categorical():

     def fit(self, BET, X ,target):

        l =(len(BET))
        BET.reset_index(drop = True, inplace = True)
        x = BET.to_dict(orient='list')
        keys =list(x.keys())

        probability = []
        likelihood = 1
        att_prior_prob = 1
        class_prior_prob = 1
        for i in range(len(BET)):
            if keys[i] != target:
                sumx = x[target][i][6]
                sumxy = x[target][i][10]
                likelihood = likelihood*(sumxy/sumx)

                class_prior_prob = (x[target][i][6]/x[target][i][5])

                count_att = x[target][i][0]
                sumxy_att = x[target][i][1]
                att_prior_prob = att_prior_prob*(sumxy_att/count_att)

        self.post_prob = (class_prior_prob * likelihood)/att_prior_prob

        return self.post_prob

