
# coding: utf-8

# In[9]:

import math
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chisqprob
import warnings
warnings.filterwarnings('ignore')


# In[15]:

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


# In[16]:

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


# In[17]:

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


# In[18]:

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
    


# In[19]:

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



