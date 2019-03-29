import numpy as np
import scipy
import copy
from scipy import signal

class Conv:

    def __init__(self,input_image_shape,stride_shape,convolution_shape,num_kernels):#(3, 10, 14),(1,1),(3, 5, 8),4
        self.delta = 1
        self.input_shape=input_image_shape#z,y,x
        self.stride_shape=stride_shape
        self.convolution_shape=convolution_shape
        self.num_kernels=num_kernels

        if len(convolution_shape)==3:
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], convolution_shape[2])
            self.bias = np.random.rand(num_kernels)
            self.stride_row=self.stride_shape[0]
            self.stride_col=self.stride_shape[1]
            self.convolution_row_shape = convolution_shape[1]
            self.convolution_col_shape =convolution_shape[2]
            self.dim1=False
        else:
            self.weights = np.random.rand(num_kernels, convolution_shape[0], convolution_shape[1], 1)
            self.bias = np.random.rand(num_kernels)
            self.stride_row = self.stride_shape[0]
            self.stride_col = 1
            self.convolution_row_shape=convolution_shape[1]
            self.convolution_col_shape=1
            self.dim1 = True

        self.input_tensor = None
        self.error_tensor = None
        self.gradient_weights = None
        self.gradient_bias = None
        self.weightsOptimizer = None
        self.biasOptimizer = None

    def forward(self,input_tensor):#batch,z,y,x

        if self.dim1:
            self.input_tensor = input_tensor.reshape(input_tensor.shape[0], self.input_shape[0], self.input_shape[1], 1)
        else:
            self.input_tensor = input_tensor.reshape(input_tensor.shape[0], self.input_shape[0], self.input_shape[1],
                                                     self.input_shape[2])

        output_tensor=np.zeros((input_tensor.shape[0],self.num_kernels,self.input_tensor.shape[2],self.input_tensor.shape[3]))
        for ba in range(input_tensor.shape[0]):
            for i in range(self.num_kernels):
                for j in range(self.input_tensor.shape[1]):
                    output_tensor[ba, i, :, :] +=scipy.signal.correlate2d(self.input_tensor[ba,j,:,:],self.weights[i,j,:,:],'same')

        for ba in range(input_tensor.shape[0]):
            for num in range(self.num_kernels):
                for i in range(output_tensor.shape[2]):
                    for j in range(output_tensor.shape[3]):
                        output_tensor[ba, num, i, j] += self.bias[num]



        #size_first=int(np.floor((self.input_shape[1]-self.convolution_shape[1])/self.stride_shape[0]))+1
        #size_second = int(np.floor((self.input_shape[2]-self.convolution_shape[2])/self.stride_shape[1]))+1
        size_first=int(np.ceil(output_tensor.shape[2]/self.stride_row))
        size_second=int(np.ceil(output_tensor.shape[3]/self.stride_col))
        output_tensor_with_stride=np.zeros((input_tensor.shape[0],self.num_kernels,size_first,size_second))
        for ba in range(input_tensor.shape[0]):
           for i in range(self.num_kernels):
             for j in range(size_first):
                for k in range(size_second):
                    j_in_output_tensor=j*self.stride_row
                    k_in_output_tensor=k*self.stride_col
                    output_tensor_with_stride[ba,i,j,k]=output_tensor[ba,i,j_in_output_tensor,k_in_output_tensor]
        self.output_shape = np.shape(output_tensor_with_stride)
        '''
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(np.shape(self.input_tensor))
        print(self.convolution_shape[1])
        print(self.convolution_shape[2])
        print(self.stride_shape[0])
        print(self.stride_shape[1])
        print(self.output_shape)
        print("####################################")
        '''
        #reshape here
        output_tensor_with_stride= output_tensor_with_stride.reshape(output_tensor_with_stride.shape[0],output_tensor_with_stride.shape[1]*output_tensor_with_stride.shape[2]*output_tensor_with_stride.shape[3])
        return output_tensor_with_stride


    def backward(self,error_tensor):
        self.error_tensor=error_tensor.reshape(self.output_shape)

        #upsampling
        self.error_tensor_upsamp=np.zeros((self.input_tensor.shape[0],self.num_kernels,self.input_tensor.shape[2],self.input_tensor.shape[3]))
        for ba in range(self.error_tensor_upsamp.shape[0]):
            for num in range(self.error_tensor_upsamp.shape[1]):
                for i in range(self.error_tensor.shape[2]):
                    for j in range(self.error_tensor.shape[3]):
                        self.error_tensor_upsamp[ba, num, i * self.stride_row, j * self.stride_col] = self.error_tensor[ba, num, i, j]

        '''
        print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
        print(np.shape(self.input_tensor))
        print(self.convolution_shape[1])
        print(self.convolution_shape[2])
        print(self.stride_shape[0])
        print(self.stride_shape[1])
        print(np.shape(self.error_tensor))
        print(np.shape(self.error_tensor_upsamp))
        print(self.error_tensor[0,0,:,:])
        print("cccc")
        print(self.error_tensor_upsamp[0,0,:,:])
        print("################################")
        '''


        #err next layer
        error_tensor_next_layer = np.zeros(np.shape(self.input_tensor))
        for ba in range(self.error_tensor.shape[0]):#batch num
          for i in range(self.input_tensor.shape[1]):#channel num
            for j in range(self.num_kernels):
              temp=scipy.signal.convolve2d(self.error_tensor_upsamp[ba,j,:,:],self.weights[j,i,:,:],'same')#same
              error_tensor_next_layer[ba, i, :, :]+=temp

        #input pad(right)
        up_size=int(np.floor(self.convolution_col_shape/2))#(3, 5, 8)
        down_size=self.convolution_col_shape-up_size-1
        left_size=int(np.floor(self.convolution_row_shape/2))
        right_size=self.convolution_row_shape-left_size-1

        self.input_tensor_padding=np.zeros((self.input_tensor.shape[0],self.input_tensor.shape[1],self.input_tensor.shape[2]+self.convolution_row_shape-1,self.input_tensor.shape[3]+self.convolution_col_shape-1))
        for ba in range(self.input_tensor.shape[0]):
            for num in range(self.input_tensor.shape[1]):
                for i in range(self.input_tensor_padding.shape[2]):
                    for j in range(self.input_tensor_padding.shape[3]):
                        if (i>left_size-1) and (i<self.input_tensor.shape[2]+left_size):
                            if(j>up_size-1) and (j<self.input_tensor.shape[3]+up_size):
                                temp=self.input_tensor[ba,num,i-left_size,j-up_size]
                                self.input_tensor_padding[ba, num, i, j] =temp
        '''
        print("!!!!!!!!!!!!!!!!!!!!!!!!")
        print(self.input_tensor.shape[2])
        print(self.input_tensor.shape[3])
        print(self.convolution_shape[1])
        print(self.convolution_shape[2])
        print(self.input_tensor[0,1,:,:])
        print(np.shape(self.input_tensor_padding[0,1,:,:]))
        print(self.input_tensor_padding[0,1,:,:])
        print("###########################")

        '''



        for ba in range(self.error_tensor_upsamp.shape[0]):
             for i in range(self.error_tensor_upsamp.shape[1]):
                self.error_tensor_upsamp[ba,i,:,:]=self.FZ(self.error_tensor_upsamp[ba,i,:,:])

        # weights
        self.gradient_weights = np.zeros((self.weights.shape[0], self.weights.shape[1], self.weights.shape[2], self.weights.shape[3]))
        for ba in range(self.error_tensor.shape[0]):
             for i in range(self.num_kernels):
                 for j in range(self.input_tensor.shape[1]):
                     self.gradient_weights[i,j,:,:]+=scipy.signal.convolve2d(self.input_tensor_padding[ba,j,:,:],self.error_tensor_upsamp[ba,i,:,:],'valid')

        '''
        gradient_weights_mid=self.gradient_weights
        for ba in range(self.gradient_weights.shape[0]):
            for i in range(self.gradient_weights.shape[1]):
                gradient_weights_mid[ba,i,:,:]=self.gradient_weights[ba,self.gradient_weights.shape[1]-i-1,:,:]

        self.gradient_weights=gradient_weights_mid
         '''
        #bias
        self.gradient_bias=np.zeros(self.num_kernels)
        gradient_bias_mid=np.zeros((self.error_tensor.shape[0],self.error_tensor.shape[1]))
        for ba in range(self.error_tensor.shape[0]):
          for i in range(self.error_tensor.shape[1]):#bias
              gradient_bias_mid[ba,i]=np.sum(self.error_tensor[ba,i,:,:])

        temp=np.sum(gradient_bias_mid,0)
        for i in range(self.error_tensor.shape[1]):
            self.gradient_bias[i]=temp[i]


        if self.weightsOptimizer is not None:
            self.weights=self.weightsOptimizer.calculate_update(self.delta,self.weights,self.get_gradient_weights())
        #else:
           # self.weights -=self.delta*self.gradient_weights

        if self.biasOptimizer is not None:
            self.bias=self.biasOptimizer.calculate_update(self.delta,self.bias,self.get_gradient_bias())
        #else:
            #self.bias -= self.delta * self.gradient_bias

        error_tensor_next_layer=error_tensor_next_layer.reshape(error_tensor_next_layer.shape[0],error_tensor_next_layer.shape[1]*error_tensor_next_layer.shape[2]*error_tensor_next_layer.shape[3])
        return error_tensor_next_layer


    def set_optimizer(self,optimizer):
        self.weightsOptimizer = copy.deepcopy(optimizer)
        self.biasOptimizer=copy.deepcopy(optimizer)
        return

    def get_gradient_weights(self):
        return self.gradient_weights

    def get_gradient_bias(self):
        return self.gradient_bias

    def initialize(self,weights_initializer, bias_initializer):
        self.weights=weights_initializer.initialize(self.weights)
        self.bias=bias_initializer.initialize(self.bias)
        return

    #fz and FZ for flip
    def fz(self,a):
        return a[::-1]

    def FZ(self,mat):
        return np.array(self.fz(list(map(self.fz, mat))))

    def get_norm(self):
        if self.weightsOptimizer is None:
            return 0
        else:
            return self.weightsOptimizer.get_norm(self.weights)


'''
    array=([[0, 1, 2],
            [4, 5, 6],
            [8, 9, 10]])

    a2=([[0, 1],
        [4, 5]])

    result1=scipy.signal.convolve2d(array, a2,mode='valid')
    result2=scipy.signal.correlate2d(array, a2,mode='valid')
    print("------")
    print(result1)
    print("------")
    print(result2)

    array0=([[[0, 1, 2],
            [4, 5, 6],
            [8, 9, 10]],

            [[0, 1, 2],
             [4, 5, 6],
             [8, 9, 10]]])

    a20=([[[0, 1],
        [4, 5]],

        [[0, 1],
         [4, 5]]]  )

    print("------")
    result3=scipy.signal.correlate(array0, a20,mode='valid')
    print(result3)

c = np.zeros((3, 1,1))
d=np.array([
[[1,2],[3,4],[5,6]],
[[1,2],[3,4],[5,6]]
])

e=c+d.reshape(3,2,2)
print(e)

array=([[1,2,3],[4,5,6]])
print(np.sum(array))

a=np.zeros(2)
a[0]=1
a[1]=2
print(a)

a=([[[1,2],[3,4],[4,5]],[[1,2],[3,4],[4,5]]])
print(np.shape(a))
print("---")


a=np.arange(12)
h=a.reshape(3,2,2)
print(h)

g=np.zeros(np.shape(h))
print(g)

print(h.shape[-1])
print(h.shape[-2])
print(h.shape[-3])


shap=(99,1)
print(shap[0])

asd=np.random.rand(3)
print(asd)
print(asd[0])


def fz(a):
    return a[::-1]


def FZ(mat):
    return np.array(fz(list(map(fz, mat))))


A = np.arange(4).reshape((2,2))
B = FZ(A)
print(A,'\n',B)

A = np.arange(12).reshape((2,2,3))
print(A)
print("!!!!!!!!!!!!!!")
print(A[0,:])
print("!!!!!!!!!!!!!!")
print(A[0,:,:])

c = np.array([[[0, 1, 2,3],
               [4, 5, 6,7]],
               [[1, 2, 3,4],
                [5,6,7,8]]])


print(c.sum(axis=2))


Matrix = [[1, 2], [3, 4]]
print(Matrix)
Matpad=np.pad(Matrix, ((1, 2), (3, 4)), 'constant')
print(Matpad)

print("----")
convolution_shape=(2,3)
weights_shape = (1, np.prod(convolution_shape[0:]))
print(weights_shape)
weights = np.random.rand(1,6)
print(np.shape(weights))

input_tensor=np.arange(6).reshape(2,3)
output_tensor=np.zeros(np.shape(input_tensor))
print(np.shape(output_tensor))

a= np.ceil(3.1)
print(a)
output_tensor=input_tensor.reshape((3,2))
print(np.shape(output_tensor))

c=np.arange(12).reshape(3,4)
c=np.pad(c, ((2, 1), (3, 4)), 'constant')
print(c)


a=np.arange(9).reshape(3,3)
b=np.sum(a)
print(b)
print(a)
c=np.sum(a,0)
print(c)
print(c[0])

a=np.zeros(5)
print(a)

a=np.random.rand(5,2)
print(a[0,0])

a=np.arange(96).reshape(3,2,4,4)
b=np.arange(16).reshape(2,2,2,2)
c=scipy.signal.correlate2d(a[0,0,:,:],b[0,1,:,:],'same')
print(np.shape(c))
'''
'''
a=np.arange(9).reshape(3,3)
print(a)
b=0-a
print(b)
'''
