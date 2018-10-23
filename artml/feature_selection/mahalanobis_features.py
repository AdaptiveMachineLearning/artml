
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


# In[2]:
class mahalanobis_selection():
    
    def find_best_feature(self, BET_best,BET_file,target,benchmark,alpha):
        best_feature = []
        for col in BET_file.columns:
            BET_target=BET_file[[target]]
            BET_col = BET_file[[col]]
            result = pd.concat([BET_best, BET_col, BET_target], axis=1)
            try:           
                Delta = self.mahalanobis(result,target)
            except:            
                Delta = 0
            if Delta/benchmark > alpha:
                best_feature = col
                benchmark = Delta
        return best_feature
    
        
    def mahalanobis(self, result, target):

        (mean1,mean2,Beta) = self.LDA_fit_transform(result, target)
        z = np.array(mean1)-np.array(mean2)    
        Delta = np.matmul(Beta.T, z)
        return Delta
    
        
    def LDA_fit_transform(self, BET, target):

        l =(len(BET.columns))
        count_1 = (BET.loc[(target), target][0]) - (BET.loc[(target), target][1])
        count_2 = BET.loc[(target), target][1]
        mean1 = []
        mean2 = []
        c = []
        for i in range(len(BET.columns)):     
            if BET.columns[i] != target:
                mean1.append((BET.loc[BET.columns[i], (target)][1] - BET.loc[BET.columns[i], (target)][10])/(BET.loc[BET.columns[i], (target)][0]-BET.loc[BET.columns[i], (target)][6]))
                mean2.append((BET.loc[BET.columns[i], (target)][10])/BET.loc[BET.columns[i], (target)][6])
        for i in range(len(BET.columns)):        
            if BET.columns[i] != target:            
                for j in range(len(BET.columns)):                
                    if BET.columns[j] != target:                    
                        cal1 = (((BET.loc[BET.columns[i], (target)][1] - BET.loc[BET.columns[i], (target)][10])*(BET.loc[BET.columns[j], (target)][1]- BET.loc[BET.columns[j], (target)][10]))/count_1)
                        cal2 = (BET.loc[BET.columns[i], (target)][10]*BET.loc[BET.columns[j], (target)][10])/count_2
                        c.append((BET.loc[BET.columns[i],(BET.columns[j])][10] -cal1 - cal2)/(count_1+count_2-2))
        c = np.array(c)
        n = (len(BET.columns)-1)
        c = reshape(c,(n,n))
        inverse = np.linalg.inv(c)

        z = np.array(mean1)-np.array(mean2)
        Beta = np.matmul(inverse, z.T)
        return (mean1,mean2,Beta)


    
    def forward_selection(self, BET_file, target, alpha=1.01):
        BET_best = pd.DataFrame()
        best_features = []
        already_selected = []
        benchmark = 0.01   
        for i in range(len(BET_file.columns)):
            best_feature = self.find_best_feature(BET_best,BET_file,target,benchmark,alpha)
            if best_feature != []:
                best_features.append(best_feature)
            if best_feature == []:            
                break
            BET_best = pd.concat([BET_best, BET_file[[best_feature]]], axis=1)
            BET_for_new_benchmark = pd.concat([BET_best, BET_file[[target]]], axis=1)
            benchmark = self.mahalanobis(BET_for_new_benchmark,target)
            already_selected = [best_feature]
            BET_file = BET_file.drop(already_selected, axis=1)      
        return best_features
