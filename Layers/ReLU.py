import numpy as np

class ReLU:


    def __init__(self):
       self.input_size = 0
       self.output_size = 0
       self.input_tensor = np.empty((0,0))
       self.error_tensor=np.empty((0,0))

    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        output_tensor =np.maximum(0,input_tensor)
        return output_tensor

    def backward(self,error_tensor):
        self.error_tensor=error_tensor
        for i in range(np.size(self.input_tensor, 0)):
            for j in range(np.size(self.input_tensor, 1)):
                if self.input_tensor[i][j] <= 0:
                    self.error_tensor[i][j] = 0
                else:
                    self.error_tensor[i][j] = error_tensor[i][j]
        return self.error_tensor



for t in np.arange(5)[::-1]:
     print(t)






