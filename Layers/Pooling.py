import numpy as np

class Pooling:

    def __init__(self,input_image_shape,stride_shape,pooling_shape):#(2, 4, 7),(2, 2),(2, 2) # (3, 2), (2, 2)
        self.input_image_shape=input_image_shape
        self.stride_shape=stride_shape
        self.pooling_shape=pooling_shape
        self.output_tensor_shape=None
        self.error_tensor=None


    def forward(self,input_tensor):
        self.input_tensor=input_tensor.reshape(input_tensor.shape[0],self.input_image_shape[0],self.input_image_shape[1],self.input_image_shape[2])
        size_first=int(np.floor((self.input_tensor.shape[2]-self.pooling_shape[0])/self.stride_shape[0])+1)
        size_second = int(np.floor((self.input_tensor.shape[3] - self.pooling_shape[1]) / self.stride_shape[1]) + 1)
        self.output_tensor=np.zeros((self.input_tensor.shape[0],self.input_tensor.shape[1],size_first,size_second))
        self.output_index = np.zeros((self.input_tensor.shape[0],self.input_tensor.shape[1], size_first, size_second))
        for ba in range(self.input_tensor.shape[0]):
            for i in range(self.input_tensor.shape[1]):
                self.output_tensor[ba, i, :, :], self.output_index[ba, i, :, :] = self.pooling(self.input_tensor[ba, i, :, :],self.pooling_shape[0],self.stride_shape[0],self.stride_shape[1], 'max')
        '''
        for ba in range(self.output_tensor_shape[0]):
            for i in range(self.output_tensor_shape[1]):
                for j in range(self.output_tensor_shape[2]):
                    for k in range(self.output_tensor_shape[3]):
                        self.output_tensor[ba, i, j, k] += 1
        '''
        self.output_tensor_shape=np.shape(self.output_tensor)
        self.output_tensor=self.output_tensor.reshape(self.output_tensor.shape[0],self.output_tensor.shape[1]*self.output_tensor.shape[2]*self.output_tensor.shape[3])
        return self.output_tensor


    def backward(self,error_tensor):
        self.error_tensor=error_tensor.reshape(self.output_tensor_shape)
        error_tensor_next_layer=np.zeros(np.shape(self.input_tensor))
        for ba in range(self.output_tensor_shape[0]):
          for i in range(self.output_tensor_shape[1]):#num of slides
            for j in range(self.output_tensor_shape[2]):
                for k in range(self.output_tensor_shape[3]):
                    index_1=int(np.floor(self.output_index[ba,i,j,k]/self.input_tensor.shape[3]))
                    index_2=int(np.mod(self.output_index[ba,i,j,k],self.input_tensor.shape[3]))
                    error_tensor_next_layer[ba, i, index_1, index_2]+=self.error_tensor[ba,i,j,k]
        '''

        for ba in range(self.output_tensor_shape[0]):
            for i in range(self.output_tensor_shape[1]):
                for j in range(self.output_tensor_shape[2]):
                    for k in range(self.output_tensor_shape[3]):
                        error_tensor_next_layer[ba, i, j, k]+=1
        '''
        error_tensor_next_layer=error_tensor_next_layer.reshape(error_tensor_next_layer.shape[0],error_tensor_next_layer.shape[1]*error_tensor_next_layer.shape[2]*error_tensor_next_layer.shape[3])
        return error_tensor_next_layer

    def pooling(self,inputMap, poolSize, poolStrideFirst,poolStrideSecond, mode='max'):
        """INPUTS:
                  inputMap - input array of the pooling layer
                  poolSize - X-size(equivalent to Y-size) of receptive field
                  poolStride - the stride size between successive pooling squares

           OUTPUTS:
                   outputMap - output array of the pooling layer

           Padding mode - 'edge'
        """
        in_row, in_col = np.shape(inputMap)
        out_row=in_row-poolSize+1
        out_col=in_col-poolSize+1
        outputMap = np.zeros((out_row, out_col))
        outputIndex = np.zeros((out_row, out_col))

        for r_idx in range(0, out_row):
            for c_idx in range(0, out_col):
                startY = r_idx
                startX = c_idx
                poolField = inputMap[startY:startY + poolSize, startX:startX + poolSize]
                poolOut = np.max(poolField)
                outputMap[r_idx, c_idx] = poolOut
                poolIndex = np.argmax(poolField)
                devide_result = int(np.floor(poolIndex / poolSize))
                reminder_result = np.mod(poolIndex, poolSize)
                real_row_index = startY + devide_result
                real_col_index = startX + reminder_result
                outputIndex[r_idx, c_idx] = real_row_index * in_col + real_col_index

        out_row_stride=int(np.ceil(out_row/poolStrideFirst))
        out_col_stride=int(np.ceil(out_col/poolStrideSecond))
        outputMap_stride = np.zeros((out_row_stride, out_col_stride))
        outputIndex_stride = np.zeros((out_row_stride, out_col_stride))

        for i in range(out_row_stride):
            for j in range(out_col_stride):
                outputMap_stride[i,j]=outputMap[i*poolStrideFirst,j*poolStrideSecond]
                outputIndex_stride[i,j]=outputIndex[i*poolStrideFirst,j*poolStrideSecond]

        return outputMap_stride,outputIndex_stride
'''
        #old
        # inputMap sizes
        in_row, in_col = np.shape(inputMap)

        # outputMap sizes
        out_row, out_col = int(np.floor(in_row / poolStrideFirst)), int(np.floor(in_col / poolStrideSecond))
        row_remainder, col_remainder = np.mod(in_row, poolStrideFirst), np.mod(in_col, poolStrideSecond)
        if row_remainder != 0:
            out_row += 1
        if col_remainder != 0:
            out_col += 1
        outputMap = np.zeros((out_row, out_col))
        outputIndex=np.zeros((out_row, out_col))
    
        # padding
        temp_map = np.lib.pad(inputMap, ((0, poolSize - row_remainder), (0, poolSize - col_remainder)), 'edge')

        # max pooling
        for r_idx in range(0, out_row):
            for c_idx in range(0, out_col):
                startY = r_idx * poolStrideFirst
                startX = c_idx * poolStrideSecond

                poolField = temp_map[startY:startY + poolSize, startX:startX + poolSize]
                poolOut = np.max(poolField)

                outputMap[r_idx, c_idx] = poolOut
                #index of max in one num form start from 0
                
                poolIndex = np.argmax(poolField)
                devide_result=int(np.floor(poolIndex / poolSize))
                reminder_result=np.mod(poolIndex, poolSize)
                real_row_index=startY+devide_result
                real_col_index=startX+reminder_result
                outputIndex[r_idx, c_idx]=real_row_index*in_row+real_col_index
    '''
        # retrun outputMap and Index









'''
a=np.array([[1,2,6,9],[7,18,1,22],[1,2,6,7]])
print(np.shape(a))
print(a)
print("------")
poolField=a[1:2,0:3]
print(poolField)
print("WWWWWWWWWWWWWWWWWWWWWWW")
b=np.max(poolField)
c=np.argmax(poolField)
e=np.max(a)
d=np.argmax(a)
print(b)
print(c)
print(e)
print(d)


data={}

data['a']=1
data['b']=2

print(data)

tu=([[2,3],[4,5]])
print(tu[1])
print(len(tu))

li=[[2,3],[4,5]]
print(li[0:2])


arr=np.array([[2,3],[4,5]])
print(arr.shape[0])

print(np.size(arr,0))

print('______')

fi=np.array([[1,2,3],[2,5,4]])
print(fi.shape[1])
print(fi[0,0])
print(fi[:,1])

H = np.eye(5) - 1 / 5
print(H)
H=-0.5*H
print(H)

a=int(np.floor(5/2))
print(a)


maa=np.arange(24)
kk=maa.reshape((2,3,4))
print(kk)
print(kk.shape[0])
print(kk.shape[1])
print(kk.shape[2])
in_row, in_col = np.shape(kk[1,:,:])
print("kk")
print(in_row)
print(in_col)

print("ffffffffffffffffffffff")
a = np.arange(9).reshape((3,3))
print(a)
print(np.max(a))
a[1,0]=8
a[1,1]=8
print(a)
lo=np.where(a==np.max(a))
print(lo[0])
print(lo[1])

list=[]
for i in range(len(lo[0])):
    list.append(lo[0][i]*3+lo[1][i])
print(list)

list.append("kk")

list1=[]
for i in range(2):
    list1.append([])
    for j in range(2):
        list1[i].append(j)

print(list1)
print(list1[1][1])
list1[1][1]=[]
print(list1)
list1[1][1].append(1)
list1[1][1].append(2)
list1[1][1].append(3)
print(list1)
print(list1[1][1])
ln=list1[1][1]
print(ln[1])
outputIndex=[]
for i in range(2):
    outputIndex.append([])
    for j in range(2):
        outputIndex[i].append([])

print(outputIndex)

outputIndex[0][1].append(1)
outputIndex[0][1].append(2)
outputIndex[0][1].append(3)
print(outputIndex)


b=np.floor(7/5)
a=np.mod(7,5)
print(b)
print(a)

print("---")
temp_map=np.zeros((2,3))
temp_map[0, 2]=1
print(temp_map[0, 2])
'''