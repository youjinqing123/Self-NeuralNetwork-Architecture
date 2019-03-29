import numpy as np
import copy
from Layers import TanH,Sigmoid,FullyConnected


class RNN:


    def __init__(self,input_size, hidden_size,output_size,bptt_length):
       self.input_size=input_size
       self.hidden_size=hidden_size
       self.output_size=output_size
       self.bptt_length=bptt_length
       #self.weights_before=np.zeros((self.hidden_size+self.input_size+1,self.hidden_size))
       #self.weights_after=np.zeros((self.hidden_size+1,self.output_size))
       self.fu1=FullyConnected.FullyConnected(self.hidden_size+self.input_size,self.hidden_size)
       self.fu2=FullyConnected.FullyConnected(self.hidden_size,self.output_size)
       self.tanh=TanH.TanH()
       self.sig=Sigmoid.Sigmoid()
       self.delta = 0.1
       self.optimizer=None
       self.TBPTT=False


    def toggle_memory(self):
        self.TBPTT =True

    def set_optimizer(self,optimizer):
        self.optimizer = copy.deepcopy(optimizer)

    def initialize(self,weights_initializer, bias_initializer):
        self.fu1.initialize(weights_initializer,bias_initializer)
        self.fu2.initialize(weights_initializer, bias_initializer)

    def get_weights(self):
        return self.fu1.get_weights(),self.fu2.get_weights()

    def set_weights(self,weights):
        self.fu1.set_weights(weights[0])
        self.fu2.set_weights(weights[1])

    def get_gradient_weights(self):
        return self.fu1.get_gradient_weights(),self.fu2.get_gradient_weights()


    def forward(self,input_tensor):
        self.input_tensor=input_tensor
        self.batch_size=self.input_tensor.shape[0]
        self.hidden_tensor = np.zeros((self.batch_size + 1, self.hidden_size))
        self.output_tensor = np.zeros((self.batch_size, self.output_size))
        for i in range(self.batch_size):
            input_combine=np.hstack((self.hidden_tensor[i].reshape(-1,self.hidden_size),self.input_tensor[i].reshape(-1,self.input_size)))
            output_fu1=self.fu1.forward(input_combine)
            output_tanh=self.tanh.forward(output_fu1)
            self.hidden_tensor[i+1,:]=output_tanh
            output_fu2=self.fu2.forward(output_tanh)
            output_sig=self.sig.forward(output_fu2)
            self.output_tensor[i,:]=output_sig

        return self.output_tensor

    def backward(self,error_tensor):
        self.error_tensor=np.zeros(np.shape(self.input_tensor))
        dh=0
        dw1=0
        dw2=0
        for i in range(self.batch_size):
            dsig=self.sig.backward(error_tensor[self.batch_size-i-1,:].reshape(-1,self.output_size))
            df2=self.fu2.backward(dsig)
            dw2+=self.fu2.get_gradient_weights()

            dh=df2+dh

            dtanh=self.tanh.backward(dh)
            df1=self.fu1.backward(dtanh)
            dw1+=self.fu1.get_gradient_weights()
            #self.error_tensor[self.batch_size-i-1,:]=df1[:,self.hidden_size:].reshape(-1,self.input_size)
            #dh=df1[:,0:self.hidden_size].reshape(-1,self.hidden_size)
            self.error_tensor[self.batch_size-i-1,:]=df1[:,self.hidden_size:]
            dh=df1[:,0:self.hidden_size]

            if self.TBPTT and self.optimizer is not None:
                if ((self.batch_size - i-1) % self.bptt_length) == 0 and i != 0:
                    w_f1 = self.fu1.get_weights()
                    w_f2 = self.fu2.get_weights()

                    w_f1 = self.optimizer.calculate_update(self.delta, w_f1, dw1)
                    w_f2 = self.optimizer.calculate_update(self.delta, w_f2, dw2)

                    self.fu1.set_weights(w_f1)
                    self.fu2.set_weights(w_f2)

                    #dh = 0

                    dw1 = 0
                    dw2 = 0

        if not self.TBPTT:
            if self.optimizer is not None:
                w_f1 = self.fu1.get_weights()
                w_f2 = self.fu2.get_weights()

                w_f1 = self.optimizer.calculate_update(self.delta, w_f1, dw1)
                w_f2 = self.optimizer.calculate_update(self.delta, w_f2, dw2)

                self.fu1.set_weights(w_f1)
                self.fu2.set_weights(w_f2)


        return self.error_tensor


'''
a=np.arange(9).reshape(3,3)+1
print(a[0,:])
b=np.zeros((1,4))
c=np.ones((1,1))
b=np.hstack((b,c))
print(b)

a=np.arange(9).reshape(3,3)+1
b=np.zeros((1,4))
print("aaaaa")
print(b)
print(np.hstack((b[0],a[1,:])))
print(np.hstack((b[0],a[1])))
print("errrrr")
err=np.arange(9).reshape(3,3)+2
b=np.zeros((1,4))
print(err)
print(b[:,:4])
err[1]=b[:,:3]
print(err)
err[1,:]=b[:,:3]+1
print(err)
print("hhhhhhh")

for i in range(9):
    if ((9 - i-1) % 3) == 0 and i != 0:
        print(i)


def agg(i,j):
    return i+1,j+1

def set(w):
    s1=w[0]
    s2=w[1]
    print(s1)
    print(s2)


f1=np.random.rand(4, 3)
f2=np.random.rand(3, 2)
c=agg(f1,f2)
print(")))))))))))))))))))")
print(c)
print(c[0])
print(")))))))))))))))))))")
print(f1)
print(f2)
print("------------------------")
set(c)


for i in range(9):
    if ((9 - i-1) % 3) == 0 and i != 0:
        print(9 - i-1)


a= np.zeros(( 3, 4))
b=np.random.rand(2, 4)
print(a)
print(b)
print(a[0])
print(b[0])
c=np.hstack((a[0,:].reshape(1,-1),b[0,:].reshape(1,-1)))
print("______")
print(np.shape(c))
g=np.random.rand(2, 8)
print(g)
print("after")
g[0,:]=c
print(g)
g=g.reshape(-1,4)
print(np.shape(g))


a=1.11
b=1.11
print(a-b)
'''
