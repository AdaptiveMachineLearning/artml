
# coding: utf-8

# In[9]:

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisqprob
import warnings
warnings.filterwarnings('ignore')


# In[14]:

def chi2(BET, feature_1 , feature_2):
    
    l =(len(BET))
    BET.reset_index(drop = True, inplace = True)
    x = BET.to_dict(orient='list')
    keys =list(x.keys())
    obs_freq = {}
    exp_freq = {}
    sum_exp_freq_vertical = np.zeros(len(feature_2))
    chi2 = 0
    
    for i in range(len(feature_1)):
        obs_freq[feature_1[i]] = []
        
        for j in range(len(feature_2)): 
            col1 = (feature_1[i])
            col2 = (feature_2[j])
            sumx = x[col1][keys.index(col2)][10]
            obs_freq[feature_1[i]].append(sumx)
            
        sum_exp_freq_vertical = sum_exp_freq_vertical + np.array(obs_freq[feature_1[i]])
    total_in_contingency = sum(sum_exp_freq_vertical)
    
    for i in range(len(feature_1)):
        exp_freq[feature_1[i]] = []
        sum_exp_freq_horizontal = sum(obs_freq[feature_1[i]])      
        for j in range(len(feature_2)):            
            e = (sum_exp_freq_horizontal*sum_exp_freq_vertical[j])/total_in_contingency              
            exp_freq[feature_1[i]].append(e)
        
    for i in range(len(feature_1)):
        for j in range(len(feature_2)):
            chi2 = chi2 + ((obs_freq[feature_1[i]][j] - exp_freq[feature_1[i]][j])**2)/exp_freq[feature_1[i]][j]
            
            
    df = (len(feature_1) - 1)*(len(feature_2)-1)
    
    print('chi2: ' + str(chi2))
    print('df: '  + str(df))
    print('chisqprob: ' + str(chisqprob(chi2, df)))
    return(chisqprob(chi2, df))


# In[ ]:



