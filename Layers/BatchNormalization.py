import numpy as np
import copy
from Layers import Base
from Layers import Initializers

class BatchNormalization:

    def __init__(self,channels=0):
        if channels > 0:
            self.weights = np.zeros((1,channels))
            self.bias = np.zeros((1,channels))
            self.mean = np.zeros((1, channels))
            self.var = np.zeros((1, channels)) + 1
        else:
            self.weights = 0
            self.bias = 0
            self.mean=0
            self.var=1

        self.batch_size = 0
        self.channels=channels
        self.dim=0
        #self.mean=0
        #self.var=1
        self.phase=Base.Phase.train
        self.weightsOptimizer = None
        self.biasOptimizer = None
        self.delta=0.1
        self.iter_num=0
        self.epislon = 1e-18

    def forward(self,input_tensor):
        if self.channels>0:
            if self.iter_num == 0:
                self.mean = np.zeros((1, self.channels))
                self.var = np.zeros((1, self.channels)) + 1
                self.weights = np.zeros((1, self.channels))+1
                self.bias = np.zeros((1, self.channels))
                self.iter_num=1+self.iter_num


            #self.dim = np.sqrt(error_tensor.shape[0] * error_tensor.shape[1] / (self.batch_size * self.channels))
            #tensor.reshape(np.int(tensor.shape[1]/channels) * tensor.shape[0], channels)
            self.input_tensor=input_tensor
            self.input_tensor=self.input_tensor.reshape(np.int(input_tensor.shape[1]/self.channels) * self.input_tensor.shape[0], self.channels)
            self.batch_size = self.input_tensor.shape[0]

            self.mean_current = np.mean(self.input_tensor, axis=0)
            self.var_current = np.var(self.input_tensor, axis=0)

            if self.phase == Base.Phase.train:
                if self.iter_num==1:
                    self.mean=self.mean_current
                    self.var=self.var_current
                    self.iter_num=self.iter_num+1
                else:
                    self.mean = 0.2 * self.mean + 0.8 * self.mean_current
                    self.var = 0.2 * np.sqrt(self.var) + 0.8 * np.sqrt(self.var_current)
                    self.var=self.var*self.var


            if self.phase == Base.Phase.train:
                self.input_tensor_hat = (self.input_tensor - self.mean_current) / np.sqrt(self.var_current + self.epislon)
            else:
                self.input_tensor_hat = (self.input_tensor - self.mean) / np.sqrt(self.var + self.epislon)

            output_tensor = self.weights * self.input_tensor_hat + self.bias
            # wx+b

            output_tensor=output_tensor.reshape(np.shape(input_tensor))
        else:
            if self.iter_num == 0:
                self.mean = np.zeros((1, input_tensor.shape[1]))
                self.var = np.zeros((1, input_tensor.shape[1]))+1

                self.weights= np.zeros((1,input_tensor.shape[1]))+1
                self.bias=np.zeros((1,input_tensor.shape[1]))
                self.iter_num=1+self.iter_num

            self.batch_size=input_tensor.shape[0]
            self.input_tensor=input_tensor
            self.mean_current=np.mean(self.input_tensor, axis=0)
            self.var_current= np.var(self.input_tensor, axis=0)
            self.mean_current = self.mean_current.reshape(1, input_tensor.shape[1])
            self.var_current = self.var_current.reshape(1,input_tensor.shape[1])

            if self.phase==Base.Phase.train:
                if self.iter_num==1:
                    self.mean=self.mean_current
                    self.var=self.var_current
                    self.iter_num=self.iter_num+1
                else:
                    self.mean = 0.2 * self.mean + 0.8 * self.mean_current
                    self.var = 0.2 * np.sqrt(self.var) + 0.8 * np.sqrt(self.var_current)
                    self.var = self.var * self.var


            if self.phase==Base.Phase.train:
                self.input_tensor_hat=(self.input_tensor-self.mean_current)/np.sqrt(self.var_current+self.epislon)
            else:
                self.input_tensor_hat = (self.input_tensor - self.mean) / np.sqrt(self.var+ self.epislon)


            output_tensor=self.weights*self.input_tensor_hat + self.bias

        return output_tensor


    def backward(self,error_tensor):
        if self.channels>0:

            self.error_tensor = error_tensor
            self.error_tensor = self.error_tensor.reshape(-1,self.channels)

            gradient_weights = 0
            for i in range(self.error_tensor.shape[0]):
                gradient_weights += self.error_tensor[i, :] * self.input_tensor_hat[i, :]
            gradient_weights = gradient_weights.reshape(np.shape(self.weights))

            gradient_bias = 0
            for i in range(self.error_tensor.shape[0]):
                gradient_bias += self.error_tensor[i, :]
            gradient_bias = gradient_bias.reshape(np.shape(self.bias))

            # err_next_layer
            gradient_input_hat = self.error_tensor * self.weights
            gradient_var = 0
            for i in range(self.error_tensor.shape[0]):
                gradient_var += self.error_tensor[i, :] * self.weights * (
                    self.input_tensor[i, :] - self.mean_current) * (-0.5) * np.power((self.var_current + self.epislon),
                                                                                     -1.5)

            gradient_mean = 0
            for i in range(self.error_tensor.shape[0]):
                gradient_mean += self.error_tensor[i, :] * self.weights * (-1) / np.sqrt(self.var_current + self.epislon)
            error_tensor_next_layer = gradient_input_hat * 1 / np.sqrt(
                self.var_current + self.epislon) + gradient_var * 2 * (
                self.input_tensor - self.mean_current) / self.batch_size + gradient_mean / self.batch_size

            if self.weightsOptimizer is not None:
                self.weights = self.weightsOptimizer.calculate_update(self.delta, self.weights, gradient_weights)
                # else:
                # self.weights -=self.delta*self.gradient_weights

            if self.biasOptimizer is not None:
                self.bias = self.biasOptimizer.calculate_update(self.delta, self.bias, gradient_bias)

            self.gradient_weights = gradient_weights
            self.gradient_bias = gradient_bias
            error_tensor_next_layer=error_tensor_next_layer.reshape(np.shape(error_tensor))

        else:
            gradient_weights=np.zeros((1,error_tensor.shape[1]))
            for i in range(error_tensor.shape[0]):
                for j in range(error_tensor.shape[1]):
                     gradient_weights[0,j]+=error_tensor[i,j]*self.input_tensor_hat[i,j]
            #gradient_weights = gradient_weights.reshape(np.shape(self.weights))

            gradient_bias=0
            for i in range(error_tensor.shape[0]):
                gradient_bias += error_tensor[i, :]
            gradient_bias = gradient_bias.reshape(np.shape(self.bias))

            # err_next_layer
            gradient_input_hat = error_tensor*self.weights
            gradient_var = 0
            for i in range(error_tensor.shape[0]):
                gradient_var += gradient_input_hat[i,:]* (self.input_tensor[i, :] - self.mean_current) * (-0.5) * np.power((self.var_current + self.epislon), -1.5)

            gradient_mean = 0
            for i in range(error_tensor.shape[0]):
                gradient_mean += gradient_input_hat[i,:] * (-1) / np.sqrt(self.var_current + self.epislon)


            error_tensor_next_layer = gradient_input_hat / np.sqrt(self.var_current+self.epislon) + gradient_var * 2 * (self.input_tensor - self.mean_current) / self.batch_size + gradient_mean / self.batch_size

            if self.weightsOptimizer is not None:
                self.weights = self.weightsOptimizer.calculate_update(self.delta, self.weights,gradient_weights)

            if self.biasOptimizer is not None:
                self.bias = self.biasOptimizer.calculate_update(self.delta, self.bias, gradient_bias)

            self.gradient_weights = gradient_weights
            self.gradient_bias = gradient_bias

        return error_tensor_next_layer


    def initialize(self,weights_initializer, bias_initializer):
        self.weights=weights_initializer.initialize(self.weights)
        self.bias=bias_initializer.initialize(self.bias)
        return

    def set_optimizer(self,optimizer):
        self.weightsOptimizer = copy.deepcopy(optimizer)
        self.biasOptimizer=copy.deepcopy(optimizer)
        return

    def get_gradient_weights(self):
        return self.gradient_weights

    def get_gradient_bias(self):
        return self.gradient_bias
