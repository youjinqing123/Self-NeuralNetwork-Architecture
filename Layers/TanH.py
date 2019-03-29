import numpy as np

class TanH:


    def __init__(self):
       pass

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        numerator=np.exp(self.input_tensor)-np.exp(-self.input_tensor)
        denominator=np.exp(self.input_tensor)+np.exp(-self.input_tensor)
        output_tensor =np.divide(numerator,denominator)
        self.derivative=1-output_tensor*output_tensor
        return output_tensor

    def backward(self,error_tensor):
        self.error_tensor=error_tensor
        self.error_tensor*=self.derivative
        return self.error_tensor









