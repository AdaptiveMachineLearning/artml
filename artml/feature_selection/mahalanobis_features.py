
# Importing all the required libraries
import os
import math
from numpy import *
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

'''

mahalanobis_selection.forward_selection is a feature selection function to select the best features that contributes to the
classifier performance. This is a real time forward selection technique which begins with no variables in the LDA model.
For each variable, the forward method calculates Δ2 (mahalanobis distance) statistics that reflect the variable's
contribution to the model if it is included.

Parameters
----------

BET_file: Input BET table. (Make sure that the index of BET is same as column names)

Target: Target variable of the classification

alpha: It is the hyperparameter for the feature selection technique. This dictates the output number of features. Default
value is 1.01


'''
class mahalanobis_selection():

    def find_best_feature(self, BET_best,BET_file,master_keys,target,benchmark,alpha):

        best_feature = []
        for col in BET_file.columns:
            columns = []
            BET_target = BET_file[[target]]
            BET_col = BET_file[[col]]
            columns = list(BET_best.columns)
            columns.append(col)
            columns.append(target)
            #Selecting the BET for particular columns & Target
            result = pd.concat([BET_best, BET_col, BET_target], axis=1)
            selected_rows = columns
            result = result.loc[selected_rows]
            result.index = list(result.columns)

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
        c = np.reshape(c,(n,n))

        try:
            inverse = np.linalg.inv(c)
        except:
            print('Handling zero determinent Exception with dummies!')
            dummies_ = np.random.random((l-1,l-1))/10000000
            inverse = np.linalg.inv(c + dummies_)

        z = np.array(mean1)-np.array(mean2)
        Beta = np.matmul(inverse, z.T)
        return (mean1,mean2,Beta)



    def forward_selection(self, BET_file, target, alpha=1.01):
        BET_best = pd.DataFrame()
        best_features = []
        already_selected = []
        benchmark = 0.0001
        master_keys = BET_file.columns
        for i in range(len(BET_file.columns)):
            best_feature = self.find_best_feature(BET_best,BET_file,master_keys,target,benchmark,alpha)
            if best_feature != []:
                best_features.append(best_feature)
            if best_feature == []:
                break
            BET_best = pd.concat([BET_best, BET_file[[best_feature]]], axis=1)
            BET_for_new_benchmark = pd.concat([BET_best, BET_file[[target]]], axis=1)


            selected_rows = list(BET_for_new_benchmark.columns)
            BET_for_new_benchmark= BET_for_new_benchmark.loc[selected_rows]
            BET_for_new_benchmark.index = list(BET_for_new_benchmark.columns)

            benchmark = self.mahalanobis(BET_for_new_benchmark,target)
            already_selected = [best_feature]

            BET_file = BET_file.drop(already_selected, axis=1)
        return best_features
