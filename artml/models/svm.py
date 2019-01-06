
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
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')


# In[11]:

class LinearSVC():
        
    def fit(self, BET, *targets , c=BET.iloc[0,0][0]):
        l =(len(BET))
        BET.reset_index(drop = True, inplace = True)
        x = BET.to_dict(orient='list')
        keys =list(x.keys())

        BET_nontargets = BET
        last_col = []
        Ee_matrix = np.zeros(shape=(l,l))

        for i in range(l):
            Ee_matrix[i] = BET_nontargets.apply(lambda x: x[i][10])
            last_col = (BET_nontargets.apply(lambda x: x[0][6]))

        for target in targets:
            
            Ee_matrix = np.array([np.delete(row, [keys.index(target) for target in targets]) for row in Ee_matrix])           
          
        Ee_matrix = np.delete(Ee_matrix, [keys.index(target) for target in targets], axis=0)
        
        last_col = (np.delete(np.array(last_col), [keys.index(target) for target in targets]))

        Ee_matrix = np.insert(Ee_matrix, len(Ee_matrix), -last_col, axis=1)

        last_col = np.append(-last_col, BET.iloc[0,0][0])

        Ee_matrix = np.insert(Ee_matrix, len(Ee_matrix), last_col , axis=0)

        Ede = np.array([[row[10] for row in x[target]] for target in targets])
        Ede_last_col = np.array([x[target][keys.index(target)][6] for target in targets])
       

        Ede_matrix_final = []
        parameters_ = []
        Ede = np.array([np.delete(row, [keys.index(target) for target in targets]) for row in Ede])

        Ede_matrix = np.insert(Ede, len(Ede)+1, -Ede_last_col, axis=1)   

        Ede_matrix_final = (2*Ede_matrix + last_col)

        I = np.identity(len(Ee_matrix))
        
        const = (((I/c)+ Ee_matrix))
        inverse = np.linalg.inv(const)

        for matrix in Ede_matrix_final:
            params_ = np.dot(inverse, np.array(matrix.T))
            parameters_.append((params_))
        self.Beta = parameters_
        
        return self.Beta
    
    def predict(self, X):
        q = []           
        numpy_matrix = X.as_matrix()
        for i in range(len(numpy_matrix)):
            result = []  
            for beta in self.Beta:
                z = np.dot(numpy_matrix[i], beta[:-1]) - beta[-1]                 
                result.append(z)
            q.append(np.argmax(result))                
        return q
    
    
    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)


# In[12]:

class LinearSVR():
    
    def fit(self, BET, target,c = 0.1):
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
        I = np.identity(n)
        const = (((I/c)+ final))

        inverse = np.linalg.inv(const)
        self.Beta = np.dot(inverse, np.array(Ede))

        return(self.Beta)
        
    def predict(self, X):
        numpy_matrix = X.as_matrix()
        result = [] 
        for i in range(len(numpy_matrix)):
            z = np.dot(numpy_matrix[i], self.Beta[:-1]) - self.Beta[-1]  
            result.append(z)
        return result
