
# coding: utf-8

# In[9]:

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisqprob
import warnings
warnings.filterwarnings('ignore')


# In[10]:

def covariance(BET):
    
    """
    This function computes pairwise covariance of all features in BET. Covariance describes 
    the linear relationship between two features.
    
    Examples
    --------
        Covariance(Basic_Element_Table)
        
        The above function generates pairwise Covariance for all the features in the Basic_Element_Table.
        
        function returns Covariance as Pandas Dataframe.
    
    """
    
    l =(len(BET))
    BET.reset_index(drop = True, inplace = True)
    x = BET.to_dict(orient='list')
    keys =list(x.keys())  
    covar = {}
    
    for i in range(len(BET)):
        covar[i] = []
        for j in range(len(BET)):
            m = keys[i]
            try:
                cov = (x[m][j][10]-(((x[m][j][1])*(x[m][j][6]))/(x[m][j][0])))/(x[m][j][0])
                covar[i].append(cov)
            except:
                covar[i].append('NaN')
            
    result = pd.DataFrame(covar, index=keys)
    result.columns = keys
    return(result)


# In[ ]:



