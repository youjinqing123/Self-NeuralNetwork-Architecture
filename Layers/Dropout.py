import numpy as np
import random
from Layers import Base



class Dropout:

    def __init__(self,probability):
        self.prob=probability
        self.fix=None
        self.phase=Base.Phase.train

    def forward(self,input_tensor):
        shape=np.shape(input_tensor)
        self.fix=np.zeros(shape)
        for i in range(self.fix.shape[0]):
            for j in range(self.fix.shape[1]):
                num=random.random()
                if num <= self.prob:
                    self.fix[i,j]=1
        if self.phase is Base.Phase.train:
            output_tensor=self.fix*input_tensor*(1/self.prob)
        else:
            output_tensor=input_tensor
        return output_tensor

    def backward(self,error_tensor):
        error_tensor_next_layer=error_tensor
        for i in range(self.fix.shape[0]):
            for j in range(self.fix.shape[1]):
                if self.fix[i,j]==0:
                    error_tensor_next_layer[i,j]=0
        if self.phase is Base.Phase.train:
            return error_tensor_next_layer
        else:
            return error_tensor




'''
a=Dropout(0.5)
print(a.phase)
if a.phase == Base.Phase.train:
    print(999)
'''