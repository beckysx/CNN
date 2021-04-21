import numpy as np
class Layer:
    def forward(self,X):
        pass
    
    def backward(self,E):
        pass

class Dense(Layer):
    def __init__(self,input_size, output_size):
        self.w = np.random.standard_normal((input_size+1, output_size))
    def forward(self,X):
        # add bias
        X = np.hstack((X,np.ones((len(X),1))))
        self.X = X
        return X @ self.w
    
    def backward(self,E):
        prev_error = E @ self.w[:-1].T
        self.w -= self.X.T @ E
        return prev_error

class Relu(Layer):
    def forward(self,X):
        self.derivative = X > 0
        # fancy index  
#         X[X<0] = 0
        return X * (X>0)
    def backward(self,E):
        return E * self.derivative
    
class SoftMax(Layer):
    def forward(self,X):
        X_exp = np.exp(X)
        X_exp_sum = np.sum(X_exp,axis=1).reshape(-1,1)
        return X_exp / X_exp_sum
    # assume we have the matching loss
    # then the error for last layer would
    # just be y - t
    def backward(self,E):
        return E

class Conv2D(Layer):
    def __init__(self,in_channel, out_channel, filter_height, filter_width, stride, padding=(0,0)):
        self.X = None
        self.inchannel = in_channel
        self.out_channel = out_channel
        self.filter_height = filter_height
        self.filter_width = filter_width
        self.stride = stride
        self.paddingh = padding[0]
        self.paddingw  = padding[1]
        
        self.filters = np.random.standard_normal((out_channel, in_channel, filter_height, filter_width))
        self.biases = np.random.standard_normal((out_channel,))
    
    def forward(self, X):
        '''
            X : input. Assume X has shape (N,C,H,W)
                N: # of sample
                C: # of in_channel
                H: image height
                W: image width
        '''
        # add padding
        X = np.pad(X,((0,0),(0,0),(0,self.paddingh),(0,self.paddingw)),'constant',constant_values=(0, 0))
        N,C,H,W = X.shape
        self.X = X
        # compue output shape
        H_o = (H + self.paddingh - self.filter_height) // self.stride + 1
        W_o = (W + self.paddingw - self.filter_width) // self.stride + 1
        # preallocate the output
        output = np.zeros((N,self.out_channel,H_o,W_o))
        
        for n in range(N): # process each image
            for f in range(self.out_channel): # iterate through each filter
                    image = X[n]
                    for i in range(H_o): # each stride down
                        for j in range(W_o): # each stride left
                            height_start = i * self.stride
                            width_start = j * self.stride
                            region = image[:,height_start:height_start+self.filter_height,width_start:width_start + self.filter_width]
                            output[n][f][i][j] += np.sum(region * self.filters[f]) + self.biases[f]
        return output
    
    def backward(self,E):
        '''
            E:  error backpropagated from previous layer.
                Assume E has shape (N, C_i, H, W) 
                N: # of sample
                C_i: # of in_channel
                H: image height
                W: image width
            
        '''
        N,C_i,H,W = E.shape
        dw = np.zeros(self.filters.shape)
        db = np.zeros(self.biases.shape)
        dx = np.zeros(self.X.shape)
        for n in range(N):
            for c in range(C_i):
                for h in range(H):
                    for w in range(W):
                        hs = h * self.stride
                        he = hs + self.filter_height
                        ws = w * self.stride
                        we = ws + self.filter_width
                        region = self.X[n,:,hs:he,ws:we]
                        e = E[n,c,h,w]
#                         print(self.filters[c].shape)
#                         print(region.shape)
                        dw[c] += region * e
                        db[c] += e
                        dx[n,:,hs:he,ws:we] += self.filters[c] * e
        self.filters -= dw
        self.biases -= db 
        return dx

                           
class MaxPooling(Layer):
    def __init__(self, kernal_size, stride=1, padding=(0,0)):
        self.kernal_size= kernal_size
        self.stride = stride
        self.padding = padding
        self.paddingh = padding[0]
        self.paddingw = padding[1]

    def forward(self,X):
        '''
            X:  input from conv layer. 
                Assume X to have shape (N, C, H, W)
                N   : # of sample
                C   : # of channel
                H   : height of the sample
                W   : width of the sample
        '''
        # add padding
        X = np.pad(X,((0,0),(0,0),(0,self.paddingh),(0,self.paddingw)),'constant',constant_values=(0, 0))
        self.shape = X.shape
        dx_indexes = []
        
        
        N,C,H,W = X.shape
        H_o = (H + self.paddingh - self.kernal_size)//self.stride + 1
        W_o = (W + self.paddingw - self.kernal_size)//self.stride + 1
        X_o = np.zeros((N,C,H_o, W_o))  # output

        for n in range(N): # iteratre through each sample
            for c in range(C): # iterate through each channel
                for h in range(H_o): # maxpooling down
                    for w in range(W_o):# maxpooling right
                        h_start = h * self.stride 
                        h_end   = h_start + self.kernal_size
                        w_start = w * self.stride
                        w_end   = w_start + self.kernal_size
                        kernal = X[n][c][h_start:h_end,w_start:w_end]
                        max_v = np.max(kernal)   # max value of current kernal
                        X_o[n][c][h][w] = max_v  # set max_val to output
                        max_i = np.argmax(kernal, axis = -1) # find max_v indx
                        dx_indexes.append([h_start + max_i[0],w_start + max_i[1]])
                        
        self.dx_indexes = dx_indexes                
        return X_o
    
    def backward(self,E):
        dx = np.zeros(self.shape)
        N,C,H_e,W_e = E.shape
        
        counter = 0    
        for n in range(N): # iteratre through each sample
            for c in range(C): # iterate through each channel
                for h in range(H_e): # down
                    for w in range(W_e): # right
                        #print(f'{n},{c},{h},{w}')
                        index = self.dx_indexes[counter]
                        #print(index)
                        dx[n][c][index[0]][index[1]] = E[n][c][h][w]
                        counter += 1             
        
        return dx

class Flatten(Layer):
    
    def forward(self, X):
        self.shape = X.shape
        X = X.reshape((self.shape[0],-1))
        return X
    
    def backward(self,E):
        E = E.reshape(self.shape)
        return E

    
def toPredict(predict):
    result = []
    for row in predict:
        max_v = 0
        r_row = []
        for x in row:
            max_v = max(max_v,x)
        for x in row:
            r_row.append(1) if x == max_v else r_row.append(0)
        result.append(r_row)
    return np.array(result)