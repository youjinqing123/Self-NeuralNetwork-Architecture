import numpy as np

class Sigmoid:


    def __init__(self):
       pass

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        denominator=1+np.exp(-self.input_tensor)
        output_tensor =np.divide(1.0,denominator)
        self.derivative=output_tensor*(1-output_tensor)
        return output_tensor

    def backward(self,error_tensor):
        self.error_tensor=error_tensor
        self.error_tensor*=self.derivative

        return self.error_tensor






