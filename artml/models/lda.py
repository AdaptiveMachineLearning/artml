
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


# In[2]:

class LinearDiscriminantAnalysis():
    
      """
        Linear Discriminant Analysis (LDA) is a classification method searching for a linear combination 
        of variables (predictors) that best separates the classes (targets). 

        It basically performs the supervised dimensionality reduction, by projecting the input data to a 
        linear subspace consisting of the directions which maximize the separation between classes (Maximizing the difference
        between the means of groups and reducing Std. deviation within groups)

        Examples
        --------
            LDA_fit(Basic_Element_Table, Target)

            where 'Basic_Element_Table' is found from BET function for the data and 'Target' is the feature that needs to be 
            predicted.

            The function returns (mean1,mean2,Beta, prob) which are Mean vectors of the groups, Linear Model coefficients and
            class probability respectively.

        """

    def fit(self, BET, target):

        l =(len(BET))
        BET1 = BET 
        BET1.reset_index(drop = True, inplace = True)
        x = BET1.to_dict(orient='list')
        keys =list(x.keys())
        k = keys.index(target)
        count_1 = BET[target][k][0] - BET[target][k][1]
        count_2 = BET[target][k][1]
        mean1 = []
        mean2 = []
        c = []
        for i in range(len(BET)):
            if i != keys.index(target):
                mean1.append((BET[target][i][1] - BET[target][i][10])/(BET[target][i][0]-BET[target][i][6]))
                mean2.append((BET[target][i][10])/BET[target][i][6])

        for i in range(len(BET)):
            if i != keys.index(target):
                for j in range(len(BET)):
                    if j != keys.index(target):
                        m = keys[i]
                        n = keys[j]
                        cal1 = (((x[m][k][6] - x[m][k][10])*(x[n][k][6] - x[n][k][10]))/count_1)
                        cal2 = (x[m][k][10]*x[n][k][10])/count_2
                        c.append((x[m][j][10]-cal1 - cal2)/(count_1+count_2-2))

        c = np.array(c)
        n = (len(BET)-1)
        c = reshape(c,(n,n))
        inverse = np.linalg.inv(c)
        z = np.array(mean1)-np.array(mean2)
        self.mean1 =mean1
        self.mean2 =mean2
        self.Beta = np.matmul(inverse, z.T)
        self.prob =  (-math.log(count_1/count_2))

        return (self.mean1,self.mean2,self.Beta,self.prob)
    
    def predict(self, X):
        numpy_matrix = X.as_matrix()
        q=[]
        for i in range(len(numpy_matrix)):
            z = numpy_matrix[i] - (0.5*(np.array(self.mean1) - np.array(self.mean2)))
            if np.matmul(self.Beta.T, z) > self.prob:
                q.append(0)
            else:
                q.append(1)
        return q


# In[ ]:



