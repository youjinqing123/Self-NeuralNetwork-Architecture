import numpy as np
from numpy import linalg as LA
import math

class L2_Regularizer:
    def __init__(self,alpha):
        self.alpha=alpha

    def calculate(self,weights):
        return self.alpha*weights

    def norm(self,weights):
        norm = 0.0
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                norm += weights[i, j]*weights[i, j]
        return self.alpha * norm



class L1_Regularizer:
    def __init__(self,alpha):
        self.alpha=alpha

    def calculate(self,weights):
        sign_we = np.zeros(np.shape(weights))
        row = weights.shape[0]
        col = weights.shape[1]
        for i in range(row):
            for j in range(col):
                if(weights[i,j]>=0):
                    sign_we[i,j]=1
                else:
                    sign_we[i, j] = -1
        return self.alpha*sign_we

    def norm(self,weights):
        norm=0.0
        for i in range(weights.shape[0]):
            for j in range(weights.shape[1]):
                norm+=np.abs(weights[i,j])
        return self.alpha*norm



