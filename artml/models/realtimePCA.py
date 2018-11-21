
# coding: utf-8

# In[3]:

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


# In[5]:

class PCA():
    def fit(self, BET): 
        """
        Principal component analysis (PCA) is a classical statistical method that uses an orthogonal transformation 
        to convert a set of observations of possibly correlated variables into a set of values of linearly uncorrelated variables 
        called principal components.

        Real time Principal components for datasets can be extracted from the ART-M covariance matrix equations.

        Examples
        --------
        PCA(Basic_Element_Table)

        This function returns eigen values & eigen vectors for the features in the Basic element table.
        """
        from artml.explore import stats
        cov = stats.covariance(BET)
        cov_mat  = cov.values
        eig_vals, eig_vecs = np.linalg.eig(cov_mat)

        print('Eigenvectors: \n%s' %eig_vecs)
        print('\nEigenvalues: \n%s' %eig_vals)

        # Make a list of (eigenvalue, eigenvector) tuples
        eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

        # Sort the (eigenvalue, eigenvector) tuples from high to low
        eig_pairs.sort(key=lambda x: x[0], reverse=True)

        # Visually confirm that the list is correctly sorted by decreasing eigenvalues
        print('\nEigenvalues in descending order:')
        for i in eig_pairs:
            print(i[0])
            


# In[ ]:



