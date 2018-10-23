
# coding: utf-8

# In[9]:

import math
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')


# In[12]:

def Ztest(BET, col1, col2):
    
    l =(len(BET))
    BET.reset_index(drop = True, inplace = True)
    x = BET.to_dict(orient='list')
    keys =list(x.keys())
    
    count = x[col2][keys.index(col1)][6]
    sumx = x[col2][keys.index(col1)][10]
    sumx2 = x[col2][keys.index(col1)][11]
    Mean = sumx/count
    Variance = (sumx2 - (((sumx)**2)/count))/count
    
    count_0 = x[col2][keys.index(col1)][0] - x[col2][keys.index(col1)][6]
    sumx_0 =  x[col2][keys.index(col1)][1] - x[col2][keys.index(col1)][10]
    sumx2_0 = x[col1][keys.index(col1)][10] -x[col2][keys.index(col1)][11]
    Mean_0 = sumx_0/count_0
    Variance_0 = (sumx2_0 - (((sumx_0)**2)/count_0))/count_0
    
    zscore = (Mean_0 - Mean)/(np.sqrt((Variance_0/count_0)+(Variance/count)))
    prob = 1 - stats.norm.cdf(zscore)
    return 2*prob


# In[ ]:



