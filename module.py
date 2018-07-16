import os
import math
from numpy import * 
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def BET(df):
    
    """ BET function constructs the Basic Element Table for the Dataframe. BET is the key step for ARTML and 
    it can be updated with the new data.
    
    BET function returns basic element table as Pandas Dataframe
    
    Notes:
    -----
    see 'Real Time Data Mining' by Prof. Sayad
    
    (https://www.researchgate.net/publication/265619432_Real_Time_Data_Mining)
    
    """
    col = df.columns.tolist()
    l = len(col)                                                              
    x ={}                                                                   # Creating empty dictionary                                 
    for m in range(l):
        for n in range(l):
            x[m,n] = []                                      # Creating keys in dictionary with empty lists
        
    for i in range(l):
        for j in range(l):
            y=col[j]
            z=col[i]
            
            """
            This code makes calculations for all the basic elements in the table. They are appended to 
            a lists of a dictionary.
            
            """
            count_x = len(df[col[i]])                                           # count in particular X column
            x[i,j].append(count_x)
            
            sum_x = df[col[i]].sum()                                                 # Sum of elemensts in y
            x[i,j].append(sum_x)
            
            sum_x2 = (df[z]*df[z]).sum()                                             # Sum of elemensts in x2
            x[i,j].append(sum_x2)
            
            sum_x3 = (df[col[i]]*df[col[i]]*df[col[i]]).sum()                        # Sum of elemensts in x3
            x[i,j].append(sum_x3)
            
            sum_x4 = (df[col[i]]*df[col[i]]*df[col[i]]*df[col[i]]).sum()             # Sum of elemensts in x4
            x[i,j].append(sum_x4)
            
            count_y = len(df[col[j]])                                          # count in particular Y column
            x[i,j].append(count_y)
            
            sum_y = df[col[j]].sum()                                                 # Sum of elemensts in y
            x[i,j].append(sum_y)
            
            sum_y2 = (df[col[j]]*df[col[j]]).sum()                                  # Sum of elemensts in y2
            x[i,j].append(sum_y2) 
            
            sum_y3 = (df[col[j]]*df[col[j]]*df[col[j]]).sum()                       # Sum of elemensts in y3
            x[i,j].append(sum_y3)
            
            sum_y4 = (df[col[j]]*df[col[j]]*df[col[j]]*df[col[j]]).sum()            # Sum of elemensts in y4
            x[i,j].append(sum_y4)
            
            sum_xy = (df[col[i]]*df[col[j]]).sum()                                  # Sum of elemensts in xy
            x[i,j].append(sum_xy)
            
            sum_xy2 = (df[col[i]]*df[col[j]]*df[col[i]]*df[col[j]]).sum()           # Sum of elemensts in (xy)2
            x[i,j].append(sum_xy2)       
            
    z={}
    for m in range(l):                                                    # converting the dictionary to DataFrame
        z[m] = []  
    for i in range(l):
        for j in range(l):
            z[i].append(x[j,i])
    result = pd.DataFrame(z, index=col)
    result.columns = col
    return(result)

def basic_elements1(x,key,e,c,i,const):
    x[key][e][0] = (x[key][e][0]+(const*1))

    x[key][e][1] = (x[key][e][1]+(const*c[i]))

    x[key][e][2] = (x[key][e][2]+(const*(c[i]*c[i])))
            
    x[key][e][3] = (x[key][e][3]+(const*(c[i]*c[i]*c[i])))
            
    x[key][e][4] = (x[key][e][4]+(const*(c[i]*c[i]*c[i]*c[i])))

    x[key][e][5] = (x[key][e][5]+(const*1))

    x[key][e][6] = (x[key][e][6]+(const*c[i]))

    x[key][e][7] = (x[key][e][7]+(const*(c[i]*c[i])))
            
    x[key][e][8] = (x[key][e][8]+(const*(c[i]*c[i]*c[i])))
            
    x[key][e][9] = (x[key][e][9]+(const*(c[i]*c[i]*c[i]*c[i])))

    x[key][e][10] = (x[key][e][10]+(const*(c[i]*c[i])))
                               
    x[key][e][11] = (x[key][e][11]+(const*(c[i]*c[i]*c[i]*c[i])))
    
    return x[key][e]


# In[9]:

def basic_elements2(x,key,k,b,c,i,m,const):    
    
    x[key][k][0] = (x[key][k][0]+(const*1))

    x[key][k][1] = (x[key][k][1]+(const*c[b.index(m)]))

    x[key][k][2] = (x[key][k][2]+(const*(c[b.index(m)]*c[b.index(m)])))
                    
    x[key][k][3] = (x[key][k][3]+(const*(c[b.index(m)]*c[b.index(m)]*c[b.index(m)])))
            
    x[key][k][4] = (x[key][k][4]+(const*(c[b.index(m)]*c[b.index(m)]*c[b.index(m)]*c[b.index(m)])))
     
    x[key][k][5] = (x[key][k][5]+(const*1))

    x[key][k][6] = (x[key][k][6]+(const*c[i]))

    x[key][k][7] = (x[key][k][7]+(const*(c[i]*c[i])))
                               
    x[key][k][8] = (x[key][k][8]+(const*(c[i]*c[i]*c[i])))
            
    x[key][k][9] = (x[key][k][9]+(const*(c[i]*c[i]*c[i]*c[i])))

    x[key][k][10] = (x[key][k][10]+(const*(c[i]*c[b.index(m)])))

    x[key][k][11] = (x[key][k][11]+(const*(c[i]*c[b.index(m)]*c[i]*c[b.index(m)])))
    
    return x[key][k]


# In[21]:

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
            basic_elements1(x,key,e,values,i,1)                           # function for updating elements  BET
            
            for m in feature_names: 
                 if m != feature_names[i]:
                    k = keys.index(m)
                    basic_elements2(x,key,k,feature_names,values,i,m,1)   # function for updating elements  BET
                    
    df = pd.DataFrame(x)
    df.index = keys
    df = df[keys]
    return df



# In[22]:

def forgetbyindex(BET, *args):
    
    BET.reset_index(drop = True, inplace = True)
    x = BET.to_dict(orient='list')
    keys = list(x.keys())
    a = [item for item in args]
    if (len(a))%2 != 0:
        print("Give correct set of Index & parameters for function")
    else:  
        b = a[0:int(len(a)/2)]
        c=  a[int(len(a)/2)::]
        for i in range(len(b)):
            key = b[i]
            e = keys.index(key)
            basic_elements1(x,key,e,c,i,-1)
            
            for m in b: 
                 if m != b[i]:
                    k = keys.index(m)
                    basic_elements2(x,key,k,b,c,i,m,-1)

    df = pd.DataFrame(x)
    df = df[keys]
    df.index = keys
    return df


# In[12]:

def growbyindex(BET, *args):
    
    main_list = list(BET.columns)
    a = [item for item in args]
    if (len(a))%2 != 0:
        print("Give correct set of Index & parameters for function")
    else:  
        b = a[0:int(len(a)/2)]
        c=  a[int(len(a)/2)::]
    
        for i in range(len(b)):
            
            elements = [[0]*12]*len(BET)
            BET[b[i]] = elements
            
            new_list = []
            for j in range(len(BET.columns)):               
                new_list.append(list(np.array([0]*12)))
    
            new_row = pd.DataFrame([new_list],columns= list(BET.columns),index = [b[i]])
            BET = pd.concat([BET,new_row])
    
        BET.reset_index(drop = True, inplace = True)
        x = BET.to_dict(orient='list')
        keys = list(x.keys())  
           
        for i in range(len(b)):
            key = b[i]
            if key in main_list:
                print('feature already exsists! Use Learn function')
            else:        
                e = keys.index(key)
                basic_elements1(x,key,e,c,i,1)

    df = pd.DataFrame(BET)
    df.index = keys
    df = df[keys]
    return df


# In[14]:

def learn(BET, df):
    col = list(df.columns)
    for index, row in df.iterrows():
        row1 = []
        for e in col:
            row1.append(row[e])
        arguments  = col + row1
        BET = learnbyindex(BET, *arguments)
    return BET


# In[16]:

def forget(BET, df):
    col = list(df.columns)
    for index, row in df.iterrows():
        row1 = []
        for e in col:
            row1.append(row[e])
        arguments  = col + row1
        BET = forgetbyindex(BET, *arguments)
    return BET


# In[18]:

def univariate(BET):                                                                                           
    
    l =(len(BET))
    BET.reset_index(drop = True, inplace = True)
    x = BET.to_dict(orient='list')
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


# In[19]:

def Covariance(BET):
    
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

	
def correlation(BET):
    
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
    
def LDA_fit(BET, target):
    l =(len(BET))
    BET1 = BET 
    BET1.reset_index(drop = True, inplace = True)
    x = BET1.to_dict(orient='list')
    keys =list(x.keys())
    k = keys.index(target)
    count_1 = BET[target][k][0] - BET[target][k][1]
    count_2 = BET[target][k][1]
    mean1 = []
    mean2 = []
    c = []
    for i in range(len(BET)):
        if i != keys.index(target):
            mean1.append((BET[target][i][1] - BET[target][i][10])/(BET[target][i][0]-BET[target][i][6]))
            mean2.append((BET[target][i][10])/BET[target][i][6])

    for i in range(len(BET)):
        if i != keys.index(target):
            for j in range(len(BET)):
                if j != keys.index(target):
                    m = keys[i]
                    n = keys[j]
                    cal1 = (((x[m][k][6] - x[m][k][10])*(x[n][k][6] - x[n][k][10]))/count_1)
                    cal2 = (x[m][k][10]*x[n][k][10])/count_2
                    c.append((x[m][j][10]-cal1 - cal2)/(count_1+count_2-2))

    c = np.array(c)
    n = (len(BET)-1)
    c = reshape(c,(n,n))
    inverse = np.linalg.inv(c)
    z = np.array(mean1)-np.array(mean2)
    Beta = np.matmul(inverse, z.T)
    prob =  (-math.log(count_1/count_2))
    
    return (mean1,mean2,Beta, prob)
    
    
def LDA_predict(train, X, target):
    (mean1,mean2,Beta, prob) = LDA_fit(BET(train), target)
    numpy_matrix = X.as_matrix()
    q=[]
    for i in range(len(numpy_matrix)):
        z = numpy_matrix[i] - (0.5*(np.array(mean1) - np.array(mean2)))
        if np.matmul(Beta.T, z) > prob:
            q.append(0)
        else:
            q.append(1)
    return q
    
def accuracy(y, y_pred):
    y = list(y)
    y_pred =list(y_pred)
    matches = []
    for i in range(len(y)):
        if y[i] == y_pred[i]:
            matches.append(1)
    return (sum(matches)/len(y))*100


def PCA(BET):
    cov = Covariance(BET)
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

def MLR(BET,target):
    
    row_indexes = list(BET.index)
    target_index = row_indexes.index(target)
    BET_features = BET.drop(target, axis =1)
    BET_features = BET_features.drop(target, axis =0)  
    cov_features = Covariance(BET_features).values
    cov_target = Covariance(BET).values
    cov_target = cov_target[target_index]
    cov_target = np.delete(cov_target, target_index)
    
    inverse = np.linalg.inv(cov_features)
    Beta_array = np.matmul(inverse, cov_target)
    
    return Beta_array	
    
