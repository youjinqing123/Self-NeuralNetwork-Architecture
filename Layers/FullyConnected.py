import numpy as np
import copy
#
class FullyConnected:

    def __init__(self,input_size,output_size):
        #print(self.input_size,self.output_size)
        self.real_input_size = input_size
        self.real_output_size = output_size
        self.weights = np.random.rand(input_size+1, output_size)
        self.optimizer = None
        self.delta=1
        #self.weights=None
        #self.bias=None
       # self.weights = np.vstack((self.weights, self.bias))


    def forward(self,input_tensor):

        self.input_tensor=np.empty((input_tensor.shape[0],input_tensor.shape[1]+1))
        for i in range(input_tensor.shape[0]):
            for j in range(input_tensor.shape[1]+1):
                if j== input_tensor.shape[1]:
                    self.input_tensor[i][j]=1.0
                else:
                    self.input_tensor[i][j]=input_tensor[i][j]
        output_tensor=np.dot(self.weights.T,self.input_tensor.T).T
        return output_tensor

    def backward(self,error_tensor):
        self.error_tensor_=error_tensor
        self.error_tensor=np.dot(error_tensor,self.weights.T)
        self.error_tensor=np.delete(self.error_tensor, -1, axis=1)
        if self.optimizer==None:
            self.weights=self.weights-0*np.dot(error_tensor.T,self.input_tensor).T
        else:
            self.weights= self.optimizer.calculate_update(self.delta, self.weights, self.get_gradient_weights())
        #self.bias=self.weights[-1,:]
        #self.weights=np.delete(self.weights,-1, axis=0)
        return self.error_tensor


    def initialize(self,weights_initializer, bias_initializer):
        self.weights=weights_initializer.initialize(np.random.rand(self.real_input_size, self.real_output_size))
        self.bias=bias_initializer.initialize(np.random.rand(1, self.real_output_size))
        self.weights=np.vstack((self.weights, self.bias))


    def set_optimizer(self,optimizer):
        self.optimizer = copy.deepcopy(optimizer)


    def get_norm(self):
        if self.optimizer is None:
            return 0
        else:
            return self.optimizer.get_norm(self.weights)

    def get_gradient_weights(self):
        return np.dot(self.error_tensor_.T,self.input_tensor).T

    def get_weights(self):
        return self.weights

    def set_weights(self,weights):
        self.weights=weights

'''

import numpy as np
from enum import Enum



class FullyConnected:
    class phase(Enum):
        pass

    def __init__(self,input_size,output_size):
        self.input_size = input_size
        self.output_size = output_size

        # weights for all
        # transpose for special memory layout
        self.weights = np.random.rand(output_size, input_size+1).T

        # learning rate
        self.delta = 1

        self.input_tensor = np.ndarray([])
        self.error_tensor = np.ndarray([])
        # error_tensor is the input of the backward phase

        self.optimizer = None

    def initialize(self, weights_initializer, bias_initializer):
        # special initializer
        # self.weights = np.delete(self.weights, -1, 1)
        weights = weights_initializer.initialize(self.weights[:-1, :])
        bias = bias_initializer.initialize(np.zeros(self.output_size))
        self.weights = np.concatenate((weights, np.expand_dims(bias, axis=0)), axis=0)
        return

    def set_optimizer(self, optimizer):
        self.optimizer = optimizer
        return

    def forward(self, input_tensor):
        # need one more line of 1
        self.input_tensor = np.concatenate((input_tensor, np.ones([input_tensor.shape[0], 1])), axis=1)
        output_tensor = self.input_tensor @ self.weights
        # print("FC forward")
        return output_tensor

    def backward(self, output_tensor):
        # update weights and return gradient to next layer
        error_tensor = output_tensor @ self.weights.T
        self.error_tensor = output_tensor
        # update
        # old version: self.weights -= self.delta * self.input_tensor.T @ output_tensor
        if self.optimizer is not None:
            self.weights = self.optimizer.calculate_update(self.delta, self.weights, self.input_tensor.T @ self.error_tensor)
        else:
            self.weights -= 0 * self.input_tensor.T @ output_tensor
        # print("FC backward")
        error_tensor = np.delete(error_tensor, -1, 1)
        return error_tensor

    def get_gradient_weights(self):
        return self.input_tensor.T @ self.error_tensor

    def get_regularization_loss(self):
        if self.optimizer.regularizer is None:
            return 0
        else:
            return self.optimizer.regularizer.norm(self.weights)

'''







