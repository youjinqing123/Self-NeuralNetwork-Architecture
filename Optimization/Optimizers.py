import numpy as np
import math

class OptimizeBase:
    def __init__(self):
        pass

    def add_regularizer(self,regularizer):
        pass


class Sgd(OptimizeBase):
    def __init__(self,global_rate):
        OptimizeBase.__init__(self)
        self.global_rate = global_rate
        self.regularizer=None

    def calculate_update(self,individual_delta, weight_tensor, gradient_tensor):
        self.weight_tensor=np.array(weight_tensor)
       # weights=self.weight_tensor.T-individual_delta*self.global_rate*gradient_tensor
        weights = self.weight_tensor - individual_delta*self.global_rate*gradient_tensor

        if self.regularizer is not None:
            weights-=individual_delta*self.global_rate*self.regularizer.calculate(weight_tensor)
        return weights

    def add_regularizer(self,regularizer):
        self.regularizer =regularizer
        return

    def get_norm(self,weights):
        weights=weights.reshape(1,-1)
        if self.regularizer is None:
            return 0
        else:
            return self.regularizer.norm(weights)


class SgdWithMomentum(OptimizeBase):
    def __init__(self,global_rate,mu):
        OptimizeBase.__init__(self)
        self.global_rate=global_rate
        self.mu=mu
        self.v=0
        self.regularizer = None
        #self.eta=mu

    def calculate_update(self,individual_delta, weight_tensor, gradient_tensor):
        self.v=self.mu*self.v-individual_delta*self.global_rate*gradient_tensor
        weights=weight_tensor+self.v

        if self.regularizer is not None:
            weights-=individual_delta*self.global_rate*self.regularizer.calculate(weight_tensor)
        return weights

    def add_regularizer(self,regularizer):
        self.regularizer =regularizer
        return

    def get_norm(self,weights):
        weights=weights.reshape(1,-1)
        if self.regularizer is None:
            return 0
        else:
            return self.regularizer.norm(weights)



class Adam(OptimizeBase):

    def __init__(self,global_rate,mu,rou,eta):
        OptimizeBase.__init__(self)
        self.global_rate = global_rate
        self.mu=mu
        self.rou=rou
        self.eta=eta
        self.k=0
        self.v = 0
        self.r = 0
        self.regularizer = None


    def calculate_update(self, individual_delta, weight_tensor, gradient_tensor):
        self.k+=1
        self.v=self.mu*self.v+(1-self.mu)*gradient_tensor
        self.r=self.rou*self.r+(1-self.rou)*gradient_tensor*gradient_tensor
        v_hat=self.v/(1-math.pow(self.mu,self.k))
        r_hat=self.r/(1-math.pow(self.rou,self.k))
        weights=weight_tensor-individual_delta*self.global_rate*(v_hat+self.eta)/(np.sqrt(r_hat)+self.eta)

        if self.regularizer is not None:
            weights-=individual_delta*self.global_rate*self.regularizer.calculate(weight_tensor)
        return weights

    def add_regularizer(self,regularizer):
        self.regularizer =regularizer
        return

    def get_norm(self,weights):
        weights=weights.reshape(1,-1)
        if self.regularizer is None:
            return 0
        else:
            return self.regularizer.norm(weights)

