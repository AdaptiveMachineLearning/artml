
# coding: utf-8

# In[1]:

import math
import numpy as np
import pandas as pd
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


# In[2]:

def univariate(BET):
    
    """
    Univariate analysis explores variables (attributes) one by one by summarizing each attribute 
    using statistical techniques. This summarizes the central tendency, dispersion and shape of 
    a dataset’s distribution, excluding NaN values.
    
    univariate Stats calculated are: ['count','Mean','Variance','Standard_deviation','coeff_of_variation','skewness','Kurtosis']
    
    Examples
    --------
        univariate(Basic_Element_Table)
        
        The above function generates Univariate statistics for all the features in the Basic_Element_Table.
        
        function returns univariate stats as Pandas Dataframe.
    
    """
    
    l =(len(BET))
    BET.reset_index(drop = True, inplace = True)
    x = BET.to_dict(orient='list')                                                 # convert BET to dictionary
    keys =list(x.keys())  
    describe = {}
    
    for i in range(l):
        describe[i] = []
        m = keys[i]
        
        try:
            count = x[m][i][0]
            describe[i].append(count)
        except:
            describe[i].append('NaN')
        try:
            Mean = (x[m][i][1])/count
            describe[i].append(Mean)   
        except:
            describe[i].append('NaN')
        
        try:
            Variance = ((x[m][i][2])-(((x[m][i][1])**2)/count))/count
            describe[i].append(Variance)
        except:
            describe[i].append('NaN')
        try:
            Standard_deviation = math.sqrt(Variance)
            describe[i].append(Standard_deviation)
        except:
            describe[i].append('NaN')
        try:
            coeff_of_variation = (Standard_deviation/Mean)*100
            describe[i].append(coeff_of_variation)
        except:
            describe[i].append('NaN')
            
        try:
            skewness = (count/((count-1)*(count-2)))*((x[m][i][3])-(3*Mean*x[m][i][2])+(3*(Mean**2)*x[m][i][1])-(count*(Mean**3)))/(Standard_deviation**3)
            describe[i].append(skewness)
        except:
            describe[i].append('NaN')
        try:
            Kurtosis = (((((count)*(count+1))/((count-1)*(count-2)*(count-3)))*((1/Standard_deviation**4)*((x[m][i][4])-(4*Mean*(x[m][i][3]))+(6*(Mean**2)*(x[m][i][2]))-(4*(Mean**3)*(x[m][i][1]))+(count*(Mean**4)))))-((3*(count-1)**2)/((count-2)*(count-3))))
            describe[i].append(Kurtosis)
        except:
            describe[i].append('NaN')        
        
    names =['count','Mean','Variance','Standard_deviation','coeff_of_variation','skewness','Kurtosis']
    result = pd.DataFrame(describe, index=names)
    result.columns = keys
    return(result)

