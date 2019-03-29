import numpy as np
import copy
from Layers import TanH,Sigmoid,FullyConnected


class LSTM:


    def __init__(self,input_size, hidden_size,output_size,bptt_length):
       self.input_size=input_size
       self.hidden_size=hidden_size
       self.output_size=output_size
       self.bptt_length=bptt_length
       self.delta = 0.1
       #self.weights_before=np.zeros((self.hidden_size+self.input_size+1,self.hidden_size))
       #self.weights_after=np.zeros((self.hidden_size+1,self.output_size))

       self.batch_size=0

       self.first=True
       self.TBPTT=False

       self.fu_f=FullyConnected.FullyConnected(self.hidden_size+self.input_size,self.hidden_size)
       self.fu_i=FullyConnected.FullyConnected(self.hidden_size+self.input_size,self.hidden_size)
       self.fu_c = FullyConnected.FullyConnected(self.hidden_size + self.input_size, self.hidden_size)
       self.fu_o = FullyConnected.FullyConnected(self.hidden_size + self.input_size, self.hidden_size)
       self.fu_final=FullyConnected.FullyConnected(self.hidden_size, self.output_size)

       self.sig_f=Sigmoid.Sigmoid()
       self.sig_i = Sigmoid.Sigmoid()
       self.tanh_c=TanH.TanH()
       self.sig_o = Sigmoid.Sigmoid()

       self.tanh_final=TanH.TanH()
       self.sig_final=Sigmoid.Sigmoid()

       self.state=(0,0)




    def toggle_memory(self):
        self.TBPTT =True

    def set_optimizer(self,optimizer):
        self.optimizer = copy.deepcopy(optimizer)

    def initialize(self, weights_initializer, bias_initializer):
        self.fu_f.initialize(weights_initializer, bias_initializer)
        self.fu_i.initialize(weights_initializer, bias_initializer)
        self.fu_c.initialize(weights_initializer, bias_initializer)
        self.fu_o.initialize(weights_initializer, bias_initializer)
        self.fu_final.initialize(weights_initializer, bias_initializer)

    def get_weights(self):
        return self.fu_f.get_weights(),self.fu_i.get_weights(),self.fu_c.get_weights(),self.fu_o.get_weights(),self.fu_final.get_weights()

    def set_weights(self,weights):
        self.fu_f.set_weights(weights[0])
        self.fu_i.set_weights(weights[1])
        self.fu_c.set_weights(weights[2])
        self.fu_o.set_weights(weights[3])
        self.fu_final.set_weights(weights[4])

    def get_gradient_weights(self):
        return self.fu_f.get_gradient_weights(),self.fu_i.get_gradient_weights(),self.fu_c.get_gradient_weights(),self.fu_o.get_gradient_weights(),self.fu_final.get_gradient_weights()



    def forward(self,input_tensor):

        self.batch_size=input_tensor.shape[0]
        self.hidden_tensor = np.zeros((self.batch_size + 1, self.hidden_size))
        self.cell_tensor=np.zeros((self.batch_size+ 1, self.output_size))

        self.input_tensor=input_tensor
        self.output_tensor = np.zeros((self.batch_size, self.output_size))




        for i in range(self.batch_size):
            input_combine=np.column_stack((self.hidden_tensor[i,:].reshape(-1,self.hidden_size), input_tensor[i,:].reshape(-1,self.input_size)))

            # f
            self.output_fu_f = self.fu_f.forward(input_combine)
            self.output_sig_f = self.sig_f.forward(self.output_fu_f)
            self.f = self.output_sig_f
            # i
            self.output_fu_i = self.fu_i.forward(input_combine)
            self.output_sig_i = self.sig_i.forward(self.output_fu_i)
            self.i = self.output_sig_i
            # c
            self.output_fu_c = self.fu_c.forward(input_combine)
            self.output_tanh_c = self.tanh_c.forward(self.output_fu_c)
            self.c = self.output_tanh_c
            # o
            self.output_fu_o = self.fu_o.forward(input_combine)
            self.output_sig_o = self.sig_o.forward(self.output_fu_o)
            self.o = self.output_sig_o

            # C
            self.cell_tensor[i+1,:]= self.cell_tensor[i,:] * self.f + self.i * self.c

            #H
            self.hidden_tensor[i+1,:]=self.o*self.tanh_final(self.cell_tensor[i+1,:])

            #Y
            self.output_fu_final=self.fu_final.forward(self.hidden_tensor[i+1,:])
            self.output_tensor[i,:]=self.sig_final.forward(self.output_fu_final)

        return self.output_tensor

    def backward(self,error_tensor):
        self.error_tensor=np.zeros(np.shape(self.input_tensor))

        #gradient_sum_prepare
        self.gradient_weight_f=0
        self.gradient_weight_i=0
        self.gradient_weight_c=0
        self.gradient_weight_o=0
        self.gradient_weight_final=0



        for i in range(self.batch_size):
            self.dh, self.dc = self.state
            # Y
            self.error_sig_final = self.sig_final.backward(error_tensor[self.batch_size - i - 1, :].reshape(-1,self.output_size))
            self.error_fu_final = self.fu_final.backward(self.error_sig_final)
            # get gradient above fu %%%%%%%%%%%%%%%%%%%%%%%%%
            self.gradient_weight_final += self.fu_final.get_gradient_weights()

            self.dh += self.error_fu_final

            # Gradient for ho in h = ho * tanh(C)
            self.error_o_before_sig = self.cell_tensor[self.batch_size - i] * self.dh
            self.error_o_after_sig = self.sig_o.backward(self.error_o_before_sig)

            # Gradient for C in h = ho * tanh(C), note we're adding dc_next here
            self.error_before_tan_final = self.dh * self.o
            self.error_after_tan_final = self.tanh_final.backward(self.error_before_tan_final)
            self.dc += self.error_after_tan_final

            # Gradient for hf in c = hf * c_old + hi * hc
            self.error_f_before_sig = self.cell_tensor[self.batch_size - i - 1] * self.dc
            self.error_f_after_sig = self.sig_f.backward(self.error_f_before_sig)

            # Gradient for hi in c = hf * c_old + hi * hc
            self.error_i_before_sig = self.c * self.dc
            self.error_i_after_sig = self.sig_i.backward(self.error_i_before_sig)

            # Gradient for hc in c = hf * c_old + hi * hc
            self.error_c_before_tanh = self.i * self.dc
            self.error_c_after_tanh = self.tanh_c.backward(self.error_c_before_tanh)

            # H and X
            self.error_f = self.fu_f.backward(self.error_f_after_sig)
            self.error_i = self.fu_i.backward(self.error_i_after_sig)
            self.error_c = self.fu_c.backward(self.error_c_after_tanh)
            self.error_o = self.fu_o.backward(self.error_o_after_sig)

            # get gradient above fu %%%%%%%%%%%%%%%%%%%%%%%%%
            self.gradient_weight_f += self.fu_f.get_gradient_weights()
            self.gradient_weight_i += self.fu_i.get_gradient_weights()
            self.gradient_weight_c += self.fu_c.get_gradient_weights()
            self.gradient_weight_o += self.fu_o.get_gradient_weights()

            # error_tensor
            error_combine = self.error_f + self.error_i + self.error_c + self.error_o
            self.error_tensor[self.batch_size - i - 1] = error_combine[:, self.hidden_size:]
            # self.hidden_tensor[self.batch_size-i-1]=error_combine[:,:self.hidden_tensor]
            self.dh = error_combine[:, :self.hidden_tensor]
            self.dc = self.dc * self.f
            self.state = (self.dh, self.dc)

            #update
            if self.TBPTT and self.optimizer is not None:
                if ((self.batch_size - i-1) % self.bptt_length) == 0 and i != 0:
                    weight_f = self.fu_f.get_weights()
                    weight_i = self.fu_i.get_weights()
                    weight_c = self.fu_c.get_weights()
                    weight_o = self.fu_o.get_weights()
                    weight_final = self.fu_final.get_weights()

                    weight_f = self.optimizer.calculate_update(self.delta, weight_f, self.gradient_weight_f)
                    weight_i = self.optimizer.calculate_update(self.delta, weight_i, self.gradient_weight_i)
                    weight_c = self.optimizer.calculate_update(self.delta, weight_c, self.gradient_weight_c)
                    weight_o = self.optimizer.calculate_update(self.delta, weight_o, self.gradient_weight_o)
                    weight_final = self.optimizer.calculate_update(self.delta, weight_final, self.gradient_weight_final)

                    self.fu_f.set_weights(weight_f)
                    self.fu_i.set_weights(weight_i)
                    self.fu_c.set_weights(weight_c)
                    self.fu_o.set_weights(weight_o)
                    self.fu_final.set_weights(weight_final)

                    #gradient reprepare
                    self.gradient_weight_f = 0
                    self.gradient_weight_i = 0
                    self.gradient_weight_c = 0
                    self.gradient_weight_o = 0
                    self.gradient_weight_final = 0



        #update
        if not self.TBPTT and self.optimizer is not None:
            weight_f = self.fu_f.get_weights()
            weight_i = self.fu_i.get_weights()
            weight_c = self.fu_c.get_weights()
            weight_o = self.fu_o.get_weights()
            weight_final = self.fu_final.get_weights()

            weight_f = self.optimizer.calculate_update(self.delta, weight_f, self.gradient_weight_f)
            weight_i = self.optimizer.calculate_update(self.delta, weight_i, self.gradient_weight_i)
            weight_c = self.optimizer.calculate_update(self.delta, weight_c, self.gradient_weight_c)
            weight_o = self.optimizer.calculate_update(self.delta, weight_o, self.gradient_weight_o)
            weight_final = self.optimizer.calculate_update(self.delta, weight_final, self.gradient_weight_final)

            self.fu_f.set_weights(weight_f)
            self.fu_i.set_weights(weight_i)
            self.fu_c.set_weights(weight_c)
            self.fu_o.set_weights(weight_o)
            self.fu_final.set_weights(weight_final)

            '''
            self.fu_f.set_optimizer(self.optimizer)
            self.fu_i.set_optimizer(self.optimizer)
            self.fu_c.set_optimizer(self.optimizer)
            self.fu_o.set_optimizer(self.optimizer)
            self.fu_final.set_optimizer(self.optimizer)
            '''
        return self.error_tensor










