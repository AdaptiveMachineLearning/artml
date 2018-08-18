
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
class mahalanobis_selection():
   
    def fit_transform(self, BET, target):
     
        count_1 = (BET.loc[(target), target][0]) - (BET.loc[(target), target][1])
        count_2 = BET.loc[(target), target][1]
        mean1 = []
        mean2 = []
        c = []

        for i in range(len(BET.columns)):     
            if BET.columns[i] != target:
                mean1.append((BET.loc[(target), BET.columns[i]][1] - BET.loc[(target), BET.columns[i]][10])/(BET.loc[(target), BET.columns[i]][0]-BET.loc[(target), BET.columns[i]][6]))
                mean2.append((BET.loc[(target), BET.columns[i]][10])/BET.loc[(target), BET.columns[i]][6])

        for i in range(len(BET.columns)):

            if BET.columns[i] != target:

                for j in range(len(BET.columns)):

                    if BET.columns[j] != target:

                        cal1 = (((BET.loc[(target), BET.columns[i]][6] - BET.loc[(target), BET.columns[i]][10])*(BET.loc[(target), BET.columns[j]][6]- BET.loc[(target), BET.columns[j]][10]))/count_1)

                        cal2 = (BET.loc[(target), BET.columns[i]][10]*BET.loc[(target), BET.columns[j]][10])/count_2
                        c.append((BET.loc[(BET.columns[j]), BET.columns[i]][10]-cal1 - cal2)/(count_1+count_2-2))

        c = np.array(c)
        n = (len(BET.columns)-1)
        c = reshape(c,(n,n))

        inverse = np.linalg.inv(c)
        z = np.array(mean1)-np.array(mean2)
        Beta = np.matmul(inverse, z.T)
        mean1 = mean1
        mean2 = mean2
        Beta = Beta
        return (mean1,mean2,Beta)
    
    def mahalanobis(self, BET, target):

        Basic_element_table = BET
        (mean1,mean2,Beta) = self.fit_transform(Basic_element_table, target)
        z = np.array(mean1)-np.array(mean2)
        Delta = np.matmul(Beta.T, z)

        return Delta
    
    def find_best_feature(self, BET_best,BET_file,target,benchmark):
        best_feature = []
        min = 1.15
        for col in BET_file.columns:
            BET_target=BET_file[target]
            BET_col = BET_file[col]
            result = pd.concat([BET_best, BET_col, BET_target], axis=1)
            try:
                Delta = self.mahalanobis(result,target)
            except:
                Delta = 0
            if Delta > (benchmark + min):
                best_feature = col
                benchmark = Delta
        return best_feature
    
    def forward_selection(self, BET_file, target):
        BET_best = pd.DataFrame()
        already_selected = []
        benchmark = 0.01
        for i in range(len(BET_file.columns)):  
            best_feature = self.find_best_feature(BET_best,BET_file,target, benchmark)
            if best_feature == []:
                break          
            BET_best = pd.concat([BET_best, BET_file[best_feature]], axis=1)
            BET_for_new_benchmark = pd.concat([BET_best, BET_file[target]], axis=1)
            self.benchmark = self.mahalanobis(BET_for_new_benchmark,target)
            print(self.benchmark)
            already_selected = [best_feature]
            BET_file = BET_file.drop(already_selected, axis=1)

        return pd.concat([BET_best, BET_file[target]], axis=1)
