import numpy as np
from enum import Enum

'''
class SoftMax:

    class phase(Enum):
        pass

    def __init__(self):
        self.input_size = 0
        self.output_size = 0
        self.loss = 0
        self.input_tensor = None
        self.error_tensor = None
        self.y_hat = None
        self.norm=None

    def forward(self, input_tensor, label_tensor):
        # print("SoftMax forward")
        # return the loss sum over batch
        self.y_hat = self.predict(input_tensor)
        loss_tensor = -np.log(self.y_hat)
        loss = np.sum(loss_tensor[label_tensor == 1])

        if self.norm is not None:
            loss += self.norm
        return loss

    def predict(self , input_tensor):
        current_input = input_tensor
        row_max = []
        sum = []
        for i in range(np.size(input_tensor, 0)):
            row_max.append(current_input[i,:].max())
       # print(row_max)
        for i in range(np.size(input_tensor, 0)):
            for j in range(np.size(input_tensor, 1)):
               current_input[i][j] = current_input[i][j]-row_max[i]
        for i in range(np.size(input_tensor, 0)):
            for j in range(np.size(input_tensor, 1)):
               current_input[i][j]=np.exp(current_input[i][j])
        for i in range(np.size(input_tensor, 0)):
            s=0.0
            for j in range(np.size(input_tensor, 1)):
                s+=current_input[i][j]
            sum.append(s)
        for i in range(np.size(input_tensor, 0)):
            for j in range(np.size(input_tensor, 1)):
                current_input[i][j]=current_input[i][j]/sum[i]
        return current_input



    def backward(self, label_tensor):
        # print("SoftMax backward: backward starts!")
        self.error_tensor = self.y_hat
        self.error_tensor[label_tensor == 1] -= 1
        return self.error_tensor


    def get_norm_from_net(self,norm):
        self.norm=norm
        return

'''


class SoftMax:
    class phase(Enum):
        pass

    def __init__(self):
        self.activation_tensor = np.ndarray([])
        self.norm = None
        # print("SoftMax initialize")

    def forward(self,input_tensor,label_tensor):
        # print("SoftMax forward")
        # return the loss sum over batch
        self.activation_tensor = self.predict(input_tensor)
        loss_tensor = -np.log(self.activation_tensor)
        loss = np.sum(loss_tensor[label_tensor == 1])



        if self.norm is not None:
            loss += self.norm
        return loss

    def backward(self,label_tensor):
        # print("SoftMax backward: backward starts!")
        error_tensor = self.activation_tensor
        error_tensor[label_tensor == 1] -= 1
        return error_tensor

    def predict(self,input_tensor):
        # print("SoftMax predcict")
        # predict y cap
        x_max = np.amax(input_tensor, axis=1)
        input_tensor = input_tensor - np.tile(x_max, (input_tensor.shape[1], 1)).T

        self.activation_tensor = np.exp(input_tensor)
        activation_sum = np.sum(self.activation_tensor, axis=1)
        self.activation_tensor = np.divide(self.activation_tensor, np.tile(activation_sum, (input_tensor.shape[1],1)).T)

        return self.activation_tensor

    def get_norm_from_net(self, norm):
        self.norm = norm
        return



