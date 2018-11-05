
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
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[4]:

"""
Bayes’ theorem provides a way of calculating the posterior probability, P(c|a), from the class (binary attribute) prior probability, 
P(c), the prior probability of the value of attribute, P(a), and the likelihood, P(a|c). 
Naive Bayes classifier assumes that the effect of the value of attribute (a) on a given class (c) is independent of the values of 
other attributes. This assumption is called class conditional independence.
"""


class GaussianNB(object):
    
    """ 
    In the real time version of Bayesian classifiers we calculate the likelihood and the prior probabilities from the 
    Basic Elements Table (BET) which can be updated in real time.
    
    """
    
    def __init__(self):
        pass

    def fit(self, BET, *targets):
        l =(len(BET))
        BET.reset_index(drop = True, inplace = True)
        x = BET.to_dict(orient='list')
        keys =list(x.keys())

        likelihood_logs = []
        for target in targets:

            count = x[target][1][6]
            class_prior_prob = (count/x[target][1][5])
            target_logs = []   
            for i in range(len(BET)):
                if keys[i] not in targets:

                    sumxy = x[target][i][10]
                    sumxy2 = x[target][i][11]
                    Mean = sumxy/count
                    Variance = (sumxy2 - (((sumxy)**2)/count))/count
                    target_logs.append(np.array([class_prior_prob,Mean,Variance]))
            likelihood_logs.append(target_logs)

        self.model = np.array(likelihood_logs)    
        print(len(self.model))
        return self

    def _prob(self, x, mean, Variance):
        exponent_ = math.exp(-(math.pow(x-mean,2)/(2*Variance)))
        return np.log(exponent_ / (np.sqrt(2 * np.pi * Variance)))
    
          
    def predict_log_proba(self, matrix):
        class_proba_ = []
        for row in matrix: 
            proba_ = []   
            for class_ in self.model:   
                i = 0
                x_proba = 0
                for feature in class_:     
                    x_proba = x_proba + (self._prob(row[i],feature[1],feature[2]))
                    i =i+1
                proba_.append(x_proba)   
            class_proba_.append(np.array(proba_))
        return class_proba_


    def predict(self, X):
        return np.argmax(self.predict_log_proba(X), axis=1)

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)

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

