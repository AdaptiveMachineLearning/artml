# Importing all the required libraries
import os
import math
from numpy import *
import numpy as np
import pandas as pd
from artml.explore import stats
import warnings
warnings.filterwarnings('ignore')


class QuadraticDiscriminantAnalysis():

    def fit(self, BET, *targets):

        l =(len(BET))
        BET1 = BET
        BET1.reset_index(drop = True, inplace = True)
        x = BET1.to_dict(orient='list')
        keys =list(x.keys())
        mean_ = []
        prob_ = []

        for target in targets:
            mu = []
            for i in range(len(BET)):
                if keys[i] not in targets:
                    mu.append((BET[target][i][10])/BET[target][i][6])
            prob_.append(math.log(BET[target][i][6]))
            mean_.append(mu)

        features = [x for x in BET1.columns if x not in targets]
        covaraince_ = stats.covariance(BET)
        BET_data = covaraince_.loc[features]
        BET_data = BET_data[features]
        covaraince_ = BET_data.as_matrix()

        self.prob_ = prob_
        self.mean_ = mean_
        self.covaraince_ = covaraince_


    def predict(self, X):
        numpy_matrix = X.as_matrix()
        q=[]
        det = np.linalg.det(self.covaraince_)
        inverse = np.linalg.inv(self.covaraince_)
        l = len(self.covaraince_)
        dummies_ = np.random.random((l,l))/10
        det_exception = np.linalg.det(self.covaraince_ + dummies_)
        exception = True
        for j in range(len(numpy_matrix)):
            result = []
            for i in range(len(self.prob_)):
                Z =  np.array(numpy_matrix[j]) - np.array(self.mean_[i])
                diff = np.matmul(inverse, Z.T)
                term1 = np.matmul(diff, Z)
                try:
                    Z_final = ((-0.5*term1)-(0.5*math.log(abs(det))) + math.log(self.prob_[i]))
                except:
                    if exception:
                        print('Handling zero determinent Exception with dummies!')
                    Z_final = ((-0.5*term1)-(0.5*math.log(abs(det_exception))) + math.log(self.prob_[i]))
                    exception = False
                result.append(Z_final)

            q.append(np.argmax(result))
        return q

    def score(self, X, y):
        return sum(self.predict(X) == y) / len(y)
