
# coding: utf-8

# In[10]:

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


# In[11]:

class SVC():
    
    def fit(self, BET, target):
        l =(len(BET))
        BET1 = BET 
        BET1.reset_index(drop = True, inplace = True)
        x = BET1.to_dict(orient='list')
        keys =list(x.keys())
        k = keys.index(target)       
        EE = []
        last_row =[]
        Ede = []
        count = BET[target][k][0]
        for i in range(len(BET)):
            if i != keys.index(target):
                for j in range(len(BET)):
                    if j != keys.index(target):
                        m = keys[i]
                        n = keys[j]
                        EE.append(x[m][j][10])
                    if j == keys.index(target):
                        Ede.append(2*(x[m][j][10]) -x[m][i][6]) 
                EE.append(-x[m][i][6])    
            last_row.append(-x[m][i][6])
        final = EE+last_row       
        final.pop()
        final.append(count)
        final = np.array(final)
        n = (len(BET))
        final = reshape(final,(n,n))

        Ede.append((count-2*(BET[target][k][1])))

        I = np.identity(n)
        const = (((I/count)+ final))

        inverse = np.linalg.inv(const)
        self.Beta = np.dot(inverse, np.array(Ede))

        return(self.Beta)

    def predict(self, X):
        numpy_matrix = X.as_matrix()
        q=[]
        intercept_ =  self.Beta.pop()
        for i in range(len(numpy_matrix)):
            
            sign = np.dot(numpy_matrix[i], self.Beta) - intercept_
            if sign > 0:
                q.append(1)
            else:
                q.append(0)
        return q


# In[12]:

class SVR():
    
    def fit(self, BET, target,tuning_parameter):
        l =(len(BET))
        BET1 = BET 
        BET1.reset_index(drop = True, inplace = True)
        x = BET1.to_dict(orient='list')
        keys =list(x.keys())
        k = keys.index(target)       
        EE = []
        last_row =[]
        Ede = []
        count = BET[target][k][0]
        for i in range(len(BET)):
            if i != keys.index(target):
                for j in range(len(BET)):
                    if j != keys.index(target):
                        m = keys[i]
                        n = keys[j]
                        EE.append(x[m][j][10])
                    if j == keys.index(target):
                        Ede.append(x[m][j][10]) 
                EE.append(-x[m][i][6])    
            last_row.append(-x[m][i][6])
        final = EE+last_row       
        final.pop()
        final.append(count)
        final = np.array(final)
        n = (len(BET))
        final = reshape(final,(n,n))

        Ede.append(-(BET[target][k][1]))
        print(Ede)
        I = np.identity(n)
        const = (((I/tuning_parameter)+ final))

        inverse = np.linalg.inv(const)
        self.Beta = np.dot(inverse, np.array(Ede))

        return(self.Beta)
        
    def predict(self, X):
        numpy_matrix = X.as_matrix()
        intercept_ = self.Beta.pop()
        return (np.dot(numpy_matrix, self.Beta) - self.intercept_)

