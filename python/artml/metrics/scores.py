
# coding: utf-8

# In[1]:

import numpy as np
import pandas as pd


# In[2]:

def accuracy(y_true, y_pred):
    y_true = list(y_true)
    y_pred =list(y_pred)
    matches = []
    for i in range(len(y_true)):
        if y_true[i] == y_pred[i]:
            matches.append(1)
    return (sum(matches)/len(y_true))*100


# In[ ]:



