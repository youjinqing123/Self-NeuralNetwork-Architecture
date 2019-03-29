import unittest
from Layers import *
import numpy as np
import NeuralNetwork
import matplotlib.pyplot as plt
import copy
import pickle


class NeuralNetwork:


    def __init__(self,optimizer,weights_initializer, bias_initializer):
        self.loss=[]
        self.layers=[]
        self.data_layer = None
        self.loss_layer = None
        self.input_tensor=None
        self.label_tensor = None
        self.weights_initializer=weights_initializer
        self.bias_initializer=bias_initializer
        self.optimizer=optimizer
        self.norm_sum=0
        self.phase=None


    def forward(self):
        '''
        input_tensor=self.layers[0].forward(self.input_tensor)
        input_tensor = self.layers[1].forward(input_tensor)
        input_tensor = self.layers[2].forward(input_tensor)
        input_tensor = self.layers[3].forward(input_tensor)
        input_tensor = self.layers[4].forward(input_tensor)
        input_tensor = self.layers[5].forward(input_tensor)
        '''
        self.norm_sum=0
        self.input_tensor ,self.label_tensor = self.data_layer.forward()
        middle_tensor=self.input_tensor
        for i in range(len(self.layers)):
            middle_tensor = self.layers[i].forward(middle_tensor)
            if isinstance(self.layers[i], FullyConnected.FullyConnected) or isinstance(self.layers[i], Conv.Conv):
                self.norm_sum+=self.layers[i].get_norm()
        self.loss_layer.get_norm_from_net(self.norm_sum)
        out=self.loss_layer.forward(middle_tensor,self.label_tensor)
        return out

    def backward(self):
        '''
        error_tensor=self.loss_layer.backward(self.label_tensor)
        error_tensor = self.layers[5].backward(error_tensor)
        error_tensor = self.layers[4].backward(error_tensor)
        error_tensor = self.layers[3].backward(error_tensor)
        error_tensor=self.layers[2].backward(error_tensor)
        error_tensor=self.layers[1].backward(error_tensor)
        self.layers[0].backward(error_tensor)
        '''
        error_tensor=self.loss_layer.backward(self.label_tensor)
        for i in range(len(self.layers)):
            error_tensor=self.layers[len(self.layers)-i-1].backward(error_tensor)
        return

    def train(self,iterations):
        for i in range(iterations):
            self.input_tensor, self.label_tensor = self.data_layer.forward()
            self.forward()
            self.backward()
            self.loss.append(self.loss_layer.loss)
        return self.loss

    def test(self,input_tensor):
        '''
        input_tensor = self.layers[0].forward(input_tensor)
        input_tensor = self.layers[1].forward(input_tensor)
        input_tensor = self.layers[2].forward(input_tensor)
        '''
        for i in range(len(self.layers)):
            input_tensor=self.layers[i].forward(input_tensor)
        return self.loss_layer.predict(input_tensor)

    def append_trainable_layer(self,layer):
        layer.initialize(copy.deepcopy(self.weights_initializer), copy.deepcopy(self.bias_initializer))
        layer.set_optimizer(copy.deepcopy(self.optimizer))
        self.layers.append(layer)
        return


    def set_phase(self,phase):
        self.phase=phase

    def set_data_layer(self,data_layer):
        self.data_layer=data_layer

    def del_data_layer(self):
        self.data_layer=None

def save(filename,net):
    f = open(filename, 'w')
    net.del_data_layer()
    pickle.dump(net, f, 0)
    f.close()
    return

def load(filename, data_layer):
    f = open(filename, 'r')
    net = pickle.load(f)
    net.set_data_layer(data_layer)
    f.close()
    return net





