
# coding: utf-8

# In[9]:

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisqprob
import warnings
warnings.filterwarnings('ignore')


# In[11]:

def correlation(BET):
    
    """
    This function computes pairwise correlations of all features in BET. correlation measures 
    how strong a relationship is between two variables.
    
    Examples
    --------
        correlation(Basic_Element_Table)
        
        The above function generates pairwise correlations for all the features in the Basic_Element_Table.
        
        function returns correlations as Pandas Dataframe.
    
    """
    
    l =(len(BET))
    BET.reset_index(drop = True, inplace = True)
    x = BET.to_dict(orient='list')
    keys =list(x.keys())  
    corr = {}
    
    for i in range(len(BET)):
        corr[i] = []
        for j in range(len(BET)):
            m = keys[i]      
            count1 = x[m][j][0]
            count2 = x[m][j][5]
            try:
                var1 = ((x[m][j][2])-(((x[m][j][1])**2)/count1))/count1
                var2 = ((x[m][j][7])-(((x[m][j][6])**2)/count2))/count2
                corrl = ((x[m][j][10]-(((x[m][j][1])*(x[m][j][6]))/(x[m][j][0])))/(x[m][j][0]))/(math.sqrt(var1*var2))
                corr[i].append(corrl)
            except:
                corr[i].append('NaN')
    
    result = pd.DataFrame(corr, index=keys)
    result.columns = keys
    return(result)


# In[ ]:



