import numpy as np
import pandas as pd
from scipy import stats
from tqdm import tqdm
from os import path, curdir

class BET(object):

    def __init__(self, gpu=False):

        self.gpu = gpu

    def create_bet(self, df):

        if self.gpu:

            """
            Need to provide additonal parameters here for installing Cuda/
            Checking Nvidia versions
            system has GPU (Nvidia. As AMD doesn't work yet)
            Add exceptions if version is not supported
            """

            """

            CUPY vectorized version code for creating BET

            """
            print('CUPY vectorized version code')

            return df

        else:


            """

            Numba CPU vectorized version code

            """

            print('Numba CPU code')

            return df
