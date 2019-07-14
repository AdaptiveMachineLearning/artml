import numpy as np
import pandas as pd

class GPU():

    def __init__(self, BET=None, feature_list=None, gpu_threshold=10000000):
        self.__BET = BET
        self.__precision = np.float64
        self._feature_list = feature_list
        self.__gpu_threshold = gpu_threshold

        
    def get_BET(self):
        """
           Returns: Basic Element Table
        """
        return self.__BET


    def set_BET(self, BET=None):
        self.__BET = BET


    def get_feature_list(self):
        return self._feature_list


    def set_feature_list(self, feature_list=None):
        self._feature_list = feature_list


    def to_dict(self, BET=None):
        return pd.DataFrame(BET.tolist(), index=self._feature_list, columns=self._feature_list)


    def __compute_BET(self, dataframe=None):
        """
           Private function to compute BET using DataFrame

           Inputs: dataframe
           Returns: Basic Element Table
        """
        input_matrix = dataframe.values.astype(np.float64)
        features = self._feature_list

        input_matrix = np.array(input_matrix)
        rows, cols = input_matrix.shape
        input_matrix_sq = np.power(input_matrix, 2)
        input_matrix_transpose = input_matrix.T
        input_matrix_sq_transpose = input_matrix_sq.T

        single_mul_arr = np.matmul(input_matrix_transpose, input_matrix)
        double_mul_arr = np.matmul(input_matrix_sq_transpose, input_matrix_sq)

        length_arr = np.full((cols,1), rows, dtype=self.__precision)
        sum_arr = np.matmul(input_matrix_transpose, np.ones((rows,1), dtype=self.__precision))
        double_arr = np.diag(single_mul_arr).reshape(cols, 1)
        triple_arr = np.diag(np.matmul(input_matrix_sq_transpose, input_matrix)).reshape(cols, 1)
        quad_arr = np.diag(double_mul_arr).reshape(cols, 1)

        stack = np.dstack(np.broadcast_arrays(length_arr, sum_arr, double_arr, triple_arr, quad_arr, length_arr.T, sum_arr.T, double_arr.T, triple_arr.T, quad_arr.T, single_mul_arr, double_mul_arr)).tolist()
        #dataframe = pd.DataFrame(stack, index=features)

        return np.array(stack)


    def create_BET(self, dataframe=None):
        self._feature_list = list(dataframe)
        rows, columns = dataframe.shape
        if rows * columns <= self.__gpu_threshold:
            self.__BET = self.__compute_BET(dataframe)
        else:
            self.__BET = np.zeros((columns, columns, 12))
            row_threshold = int(self.__gpu_threshold / columns)
            chunk_size = int(rows / row_threshold)
            idx = 0
            while chunk_size != 0:
                self.__BET = self.__BET + self.__compute_BET(dataframe.iloc[idx:idx+row_threshold,:])
                idx += row_threshold
                chunk_size -= 1
            if idx < rows:
                self.__BET = self.__BET + self.__compute_BET(dataframe.iloc[idx:rows,:])
        return self.__BET


    def incremental_learning(self, dataframe=None):
        #use the initialized bet and add incremented part
        orignal_BET = self.__BET.copy()
        self.create_BET(dataframe) #overwrites the global BET
        self.__BET = orignal_BET + self.__BET
        return self.__BET


    def decremental_learning(self, dataframe=None):
        #use the initialized bet and remove values from this dataframe
        orignal_BET = self.__BET.copy()
        self.create_BET(dataframe) #overwrites the global BET
        self.__BET = orignal_BET - self.__BET
        return self.__BET


    def add_features(self, dataframe=None):
        return self.create_BET(dataframe)
      

    def remove_features(self, feature=None):
        #remove these features from the dataframe.
        idx = self._feature_list.index(feature)
        self._feature_list.remove(feature)
        self.__BET = np.delete(self.__BET, (idx), axis=0)
        self.__BET = np.delete(self.__BET, (idx), axis=1)
        return self.__BET

