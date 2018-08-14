
# coding: utf-8

# In[9]:

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisqprob
import warnings
warnings.filterwarnings('ignore')


# In[13]:

def Ttest(BET, col1, col2):
    
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
    
    var = (((count_0-1)*Variance_0) + ((count-1)*Variance))/(count_0 + count - 2)
    
    tscore = (Mean_0 - Mean)/(np.sqrt(var*((1/count_0)+(1/count))))
    
    df = (count + count_0 - 2)
    
    prob = (1-stats.t.cdf(tscore, df)) 
    return 2*prob


# In[ ]:



