import numpy as np


class Constant:
    def __init__(self,con):
        self.con=con
    def initialize(self, weights):
        weights=np.zeros(np.shape(weights))+self.con
        return weights


class UniformRandom:
    def __init__(self):
        pass
    def initialize(self,weights):
        shape=np.shape(weights)
        num= np.prod(shape)
        weights = np.random.rand(num)
        weights=weights.reshape(shape)
        return weights

class Xavier:
    def __init__(self):
        pass
    def initialize(self, weights):
        weights=np.random.normal(0.0,np.sqrt(2.0/(weights.shape[0]+weights.shape[1])),np.shape(weights))
        return weights

'''
u1=Constant()

print(u1.initialize([[1,2],[3,4]]))

u2=Xavier()
print(u2.initialize([[1,2],[3,4]]))

u3=UniformRandom()
print(u3.initialize([[1,2],[3,4]]))
'''
'''
weights=np.vstack(([[1,2],[3,4]],[[5,6],[7,8]]))
print(weights)

weights_1=np.array([[1,2,3],[4,5,6]])
print("--")
print(weights_1)
zero=np.zeros(np.shape(weights_1))+0.1
print(zero)

norm=np.random.normal(0.0,1,np.shape(weights_1))
print(norm)

print(norm.shape[0])
print(norm.shape[1])
print(norm.shape[-1])
print(norm.shape[-2])

'''

