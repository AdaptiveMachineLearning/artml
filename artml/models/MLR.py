
# coding: utf-8

# In[2]:

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


# In[32]:

class LinearRegression():
    
    def fit(self, BET,target):
        from artml.explore import stats
        row_indexes = list(BET.index)
        target_index = row_indexes.index(target)
        BET_features = BET.drop(target, axis =1)
        BET_features = BET_features.drop(target, axis =0)  
        cov_features = stats.covariance(BET_features).values
        cov_target = stats.covariance(BET).values
        cov_target = cov_target[target_index]
        cov_target = np.delete(cov_target, target_index)
        inverse = np.linalg.inv(cov_features)
        Beta_array = np.matmul(inverse, cov_target)

        l =(len(BET))
        BET.reset_index(drop = True, inplace = True)
        x = BET.to_dict(orient='list')
        keys =list(x.keys())

        mean_target = (BET[target][keys.index(target)][1])/BET[target][keys.index(target)][0]
        mean_X = []

        for i in range(len(BET_features)):
            if i != keys.index(target):
                mean_X.append((BET[target][i][1])/BET[target][i][0])
        
        self.Beta_array = Beta_array
        self.intercept_ = mean_target - np.matmul(self.Beta_array, mean_X)

        return (self.intercept_, self.Beta_array)
    
    def predict(self, X):
        numpy_matrix = X.as_matrix()
        
        return (np.dot(numpy_matrix, self.Beta_array) + self.intercept_)


# In[ ]:



