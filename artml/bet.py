
# coding: utf-8

# In[23]:

import math
import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')


# In[24]:

def create_bet(df):

    """ BET function constructs the Basic Element Table for the Dataframe. BET is the key step for ARTML and
    it can be updated with the new data.

    BET function returns basic element table as Pandas Dataframe

    Notes:
    -----
    see 'Real Time Data Mining' by Prof. Sayad

    (https://www.researchgate.net/publication/265619432_Real_Time_Data_Mining)

    """
    col = df.columns.tolist()
    df_matrix = df.values
    l = len(col)

    idx = np.array([5,6,7,8,9,0,1,2,3,4,10,11])
    bet={}
    x = np.array([[np.zeros(12) for x in range(l)] for y in range(l)])
    for i in tqdm(range(l)):
        bet[i] = []

        for j in range(i,l):
            y= np.array(df_matrix[:,j])
            z= np.array(df_matrix[:,i])

            """
            This code makes calculations for all the basic elements in the table. They are appended to
            a lists of a dictionary.
            """
            
            x[i,j] = np.array([len(z), z.sum(), (z**2).sum(), (z**3).sum(), (z**4).sum(), 
                               len(y), y.sum(), (y**2).sum(), (y**3).sum(), (y**4).sum(), (z*y).sum(), ((z*y)**2).sum()])

            x[j,i] = x[i,j][idx]
      
        for j in range(l): 
           bet[i].append(x[j,i])

    result = pd.DataFrame(bet, index=col)
    result.columns = col
    return(result)




# In[25]:

def calculate_basic_elements1(x,key,e,c,const):
    
    """ This is an inner function used in learn_by_index & grow_by_index functions for making 
    calculations to update the BET
    
    This takes (BET_dictionary, feature_name, feature_index, values_list, i, +1/-1 (const)) as arguments 
    for making the calculations
    """
    
    array = np.array(x[key][e])
    
    array = array + const*(np.array([1,c, c**2,c**3,c**4,1,c, c**2,c**3,c**4,c**2,c**4]))
    
    x[key][e] = array
    
    return x[key][e]


# In[26]:

def calculate_basic_elements2(x,key,k,b,c,i,m,const):   

    
    """ This is an inner function used in learn_by_index & grow_by_index functions for making 
    calculations to update the BET
    
    This takes (BET_dictionary, feature_name, feature_index,feature_names_list, values_list, i, m, +1/-1 (const)) as arguments 
    for making the calculations
    """
   
    array = np.array(x[key][k])
    
    array = array + const*(np.array([1,c[b.index(m)],  (c[b.index(m)])**2,(c[b.index(m)])**3,(c[b.index(m)])**4,1,c[i], c[i]**2,c[i]**3,c[i]**4, c[i]*(c[b.index(m)]),(c[i]*(c[b.index(m)])**2)]))
    
    x[key][k] = array
    
    return x[key][k]

# In[27]:

def learnbyindex(BET, *args):
    
    """ This function takes Basic Element Table and feature_names & values as arguments to update the 
        given list of feature column & rows in the BET by corresponding values.
        
        Examples
        --------
        learnbyindex(Basic_Element_Table, 'feature_1','feature_2', 1, 2 )
        
        The above function updates feature_1, feature_2 in the BET by values 1 and 2 respectively.
    
    """
   
    BET.reset_index(drop = True, inplace = True)                               # convert BET to dictionary
    x = BET.to_dict(orient='list')
    keys = list(x.keys())
    arguments_list = [item for item in args]
    n_features = int(len(arguments_list)/2)                          # no of features given as input for updating BET
    
    if (len(arguments_list))%2 != 0:                    
        print("Error: Give correct set of Feature_names & corresponding parameters")
    
    else:  
        feature_names = arguments_list[0:n_features]
        values=  arguments_list[n_features::]
        
        for i in range(len(feature_names)):
            key = feature_names[i]
            e = keys.index(key)
            calculate_basic_elements1(x,key,e,values[i],1)                           # function for updating elements  BET
                        
            for m in feature_names: 
                 if m != feature_names[i]:
                    k = keys.index(m)
                    calculate_basic_elements2(x,key,k,feature_names,values,i,m,1)   # function for updating elements  BET
                    
    df = pd.DataFrame(x)
    df.index = keys
    df = df[keys]
    return df


# In[28]:

def forgetbyindex(BET, *args):
    
    """ This function takes Basic Element Table and feature name & values as arguments to update the 
        given list of features in the BET by corresponding values (deleting effect of those values from BET).
        
        Examples
        --------
        forgetbyindex(Basic_Element_Table, 'feature_1','feature_2', 1, 2 )
        
        The above function reduces feature_1, feature_2 in the BET by values 1 and 2 respectively.
    
    """
    
    BET.reset_index(drop = True, inplace = True)
    x = BET.to_dict(orient='list')                                                   # convert BET to dictionary
    keys = list(x.keys())
    arguments_list = [item for item in args]
    n_features = int(len(arguments_list)/2)  
    
    if (len(arguments_list))%2 != 0:                                        # no of features given as input for updating BET
        print("Give correct set of Index & parameters for function")
    else:  
        feature_names = arguments_list[0 : n_features]
        values=  arguments_list[n_features: :]
        for i in range(n_features):
            key = feature_names[i]
            e = keys.index(key)
            calculate_basic_elements1(x,key,e,values,i,-1)                                  # function for updating elements  BET
            
            for m in feature_names: 
                 if m != feature_names[i]:
                    k = keys.index(m)
                    calculate_basic_elements2(x,key,k,feature_names,values,i,m,-1)

    df = pd.DataFrame(x)
    df = df[keys]
    df.index = keys
    return df



# In[29]:


def growbyindex(BET, *args):
    
    """ This function takes Basic Element Table and feature name & values as arguments to update the 
        BET with new features and corresponding values.
        
        Examples
        --------
        growbyindex(Basic_Element_Table, 'new_feature_1','new_feature_2', 1, 2 )
        
        The above function adds new_feature_1, new_feature_2 in the BET with values 1 and 2 respectively.
    
    """
    
    main_list = list(BET.columns)
    arguments_list = [item for item in args]                                            # convert BET to dictionary
    n_features = int(len(arguments_list)/2)
    if (len(arguments_list))%2 != 0:
        print("Give correct set of Index & parameters for function")
    else:  
        feature_names = arguments_list[0:n_features]
        values =  arguments_list[n_features::]
    
        for i in range(n_features):
            
            elements = [[0]*12]*len(BET)                                         #Creating null  basic elements lists
            BET[feature_names[i]] = elements
            
            new_list = []
            for j in range(len(BET.columns)):               
                new_list.append(list(np.array([0]*12)))
    
            new_row = pd.DataFrame([new_list],columns= list(BET.columns),index = [feature_names[i]])
            BET = pd.concat([BET,new_row])
    
        BET.reset_index(drop = True, inplace = True)
        x = BET.to_dict(orient='list')
        keys = list(x.keys())  
           
        for i in range(n_features):
            key = feature_names[i]
            if key in main_list:
                print('feature already exsists! Use Learn function')
            else:        
                e = keys.index(key)
                calculate_basic_elements1(x,key,e,c,i,1)

    df = pd.DataFrame(BET)
    df.index = keys
    df = df[keys]
    return df


# In[30]:

def learn(BET, df):
          
    """ This function takes Basic Element Table and dataframe as inputs to update the 
        BET with new data in the dataframe. (Incremental Learning of BET with new dataframe as input)
        
        Examples
        --------
        learn(Basic_Element_Table, data_frame)
        
        The above function updates Basic_Element_Table with values in the new dataframe.
    
    """
    
    col = list(df.columns)
    for index, row in df.iterrows():
        row1 = []
        for e in col:
            row1.append(row[e])
        arguments  = col + row1
        BET = learnbyindex(BET, *arguments)
    return BET


# In[31]:

def forget(BET, df):
    
    """ This function takes Basic Element Table and dataframe as inputs to change and remove the  
        effect of that data in the BET. (Decremental Learning of BET with dataframe as input)
        
        Examples
        --------
        forget(Basic_Element_Table, data_frame)
        
        The above function updates Basic_Element_Table with values in the new dataframe.
    
    """
    
    col = list(df.columns)
    for index, row in df.iterrows():
        row1 = []
        for e in col:
            row1.append(row[e])
        arguments  = col + row1
        BET = forgetbyindex(BET, *arguments)
    return BET


