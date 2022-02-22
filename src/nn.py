import numpy as np
import numpy.random
import numpy.matlib
import matplotlib.pyplot as plt
from matplotlib import cm



"""
A "from-scratch" neural network implementation for educational purposes.
This version is a simple binary classififer. It allows the user to incrementally
train the network, displaying the resulting decision boundary and full density.
The training data is hard-coded, but there is also a loop to allow more
custom training data. We initialize the weights to be drawn from a standard
Gaussian distribution.
A crude diagram of a NN with two hidden layers and 2-dimensional inputs.
The final layer is returned as a weighted sum, passed through a final
sigmoid function. This scalar will determine which class to assign to the
input point.
x1--O---O
  \ / \ / \ 
   x   x   O --
  / \ / \ /
x2--O---O
@author: Joseph Anderson <jtanderson@salisbury.edu>
@date:   28 May 2019
Exercise 1: vectorize more of the operations, combine the input, output, and
hidden layers into single matrices. 
Exercise 2: Adapt the model to learn more than two classes
Exercise 3: Use "convolutional" or "recurrent" neuron architectures
Exercise 4: Turn into a "generative" model, to generate typical examples
from either of the two classes
Exercise 5: Parallelize!
For motivation/explanation, see, for example:
https://google-developers.appspot.com/machine-learning/crash-course/backprop-scroll/
"""


class NN:
    def __init__(self, d=2, nl=1, ls=2, r=.1, b=False, out=1):
        # The dimensionality of the input data
        self.dim = d
        
        # The number of hidden layers
        self.num_layers = nl
        
        # The size of each hidden layer
        self.layer_size = ls
        
        # The step size used in gradient descent
        self.rate = r
        self.bias = b
        
        #dim of the outputted vector
        self.outDim = out

        # add a dimension for bias
        if self.bias:
            self.dim += 1

        # X holds N-by-d samples
        #   - N is number of samples
        #   - d is dimension
        self.X = np.empty((0,self.dim), float)

        # Y holds N labels of -1 or 1
        self.Y = np.array([])

        # input weights. Row i is the array of weights applied to x_i
        self.w_in = np.random.standard_normal((self.dim,self.layer_size))

        # "Tensor" (3-dim array) of hidden-layer output weights. 
        # w_hidden[lay][i][j] is the weight between lay node i and lay+1 node j
        self.w_hidden = np.random.standard_normal((self.num_layers-1, self.layer_size, self.layer_size))

        # output weights, comes from last layer
        self.w_out = np.random.standard_normal((self.outDim,self.layer_size))

    # Use the standard sigmoid function. Another option is arctan, etc.
    def sigmoid(self, arr):
        return 1/(1+np.exp(-1*arr))

    # The derivative of the sigmoid function.
    # Check this by hand to see how convenient it is :)
    def sigmoid_deriv(self, arr):
        return self.sigmoid(arr) * (1 - self.sigmoid(arr))

    # The squared error between to vectors/scalars
    def msqerr(self, pred, ans):
        return np.sum((pred-ans)**2)/2

    def reset(self):
        self.w_in = np.random.standard_normal((self.dim,self.layer_size))
        self.w_hidden = np.random.standard_normal((self.num_layers-1, self.layer_size, self.layer_size))
        self.w_out = np.random.standard_normal((self.outDim,self.layer_size))





# forward_step takes the weights of the network and an input point,
# returning the scalar output of the network, along with a matrix
# which is a record of the output of each intermediate node during
# the computation. This is needed for training and verification.

# Arguments:
# w_in is the dim-by-h matrix of input weights to the first layer
# w_out is the h-by-1 array of weights from the last hidden layer to the output node
# w_hidden is the num_layers-by-layer_size-by-layer_size matrix of weights between each layer
#     hidden[i] has the weights from i to i+1
#     hidden[i][j] is the array of weights into node j of layer i+1
# data is 1-by-dim row vector

# Returns:
# scalar value coming out of the output node
# outs is layers-by-layer_size to store the output of each node

    def forward_step(self, data):
        outs = np.array([self.sigmoid(data @ self.w_in)]) # 1-by-dim times dim-by-h
        for i in range(1,self.num_layers):
            # i-1 here because w[i] is output weights
            # get the output of the last layer (sig of x) and weight it into this layer
            ins = outs[-1] @ self.w_hidden[i-1]  # 1-by-h times h-by-h
            outs = np.append(outs, [self.sigmoid(ins)], axis=0)

        # last row of outs now holds the weighted output of the last hidden layer
        ret = self.sigmoid(outs[-1] @ self.w_out.T)
        return ret, outs


# backprop analyzes how wrong the network was at predicting a given label,
# then uses the magnitude of the error to perform gradient descent on the
# edge weights throughout the network. Check this with the chain rule
# of the error function! It tracks the change in error with respect to weights,
# inputs, and outputs of every node in the network

# w_in: dim-by-layer_size
#     weights of the input nodes
# w_out: 1-by-layer_size
#     weights to the output node
# w_hidden: num_layers-1 x layer_size x layer_size
#     w_hidden[lay][i][j] is the weight between lay node i and lay+1 node j
#     a column is all input weights to that node
# outputs: num_layers x layer_size
#     record of every node's output from the forward pass
# pred: scalar predicted output
# data: the input data point
# label: scalar true output

    def backprop(self, outputs, pred, data, label):
        dEyo = pred - label # vector 1xM - 1xM (M = dim of input)
        dExo = dEyo * self.sigmoid_deriv(outputs[-1] @ self.w_out.T) # vector 1xM @ (1xL @ LxM)
        dEwo = np.array([dExo]).T @ np.array([outputs[-1]]) #Mx1 @ 1xL

        # hidden layer derivatives setup
        dEwh = np.zeros((self.num_layers-1, self.layer_size, self.layer_size))
        dExh = np.zeros((self.num_layers, self.layer_size))
        dEyh = np.zeros((self.num_layers, self.layer_size))

        # need to do output layer first, not a matrix product
        dEyh[-1] = dExo @ self.w_out  #  1xM @ MxL

        for i in range(self.num_layers-2,-1,-1):
            # i-1 to get the inputs to layer i
            x = outputs[i-1] @ self.w_hidden[i-1] # 1-by-h times h-by-h
            dExh[i] = dEyh[i] * self.sigmoid_deriv(x) # 1-by-h
            dEwh[i] = outputs[i-1] * dExh[i]
            if i > 0:
                # prep the next layer
                dEyh[i-1] = self.w_hidden[i] @ dExh[i].T # h-by-h times h-by-1

        #dEwi = outputs[0] * dEyh[0] # take care of the input layer, again
                                    # not a matrix product
        data = numpy.array([data])
        dEwi = np.matlib.repmat(data.T, 1, self.layer_size) * np.matlib.repmat(dExh[0], self.dim, 1)  # dim-by-h broadcast dim-by-h


        # adjust the hiden layer weights accoriding to the error.
        # Check to see that this follows gradient descent!
        self.w_hidden = self.w_hidden - self.rate * dEwh
        self.w_in = self.w_in - self.rate * dEwi
        self.w_out = self.w_out - self.rate * dEwo

    def train_rounds(self, train_x, train_y, num_rounds):    
        # iterate as long as we're told
        # For each epoch, it would be helpful to print the total "loss" -- the error
        # across the whole training set.
        # Often, one might choose a loss threshold (say, < 0.0001) and simply train until
        # the loss is smaller
        for i in range(1,num_rounds+1):
            #print(f"Itteration i: {i}")
            # iterate each data point
            loss = 0
            for j in range(0,train_x.shape[0]):
                dat = train_x[j]
                if self.bias:
                    dat = np.append(train_x[j], [1])

                # get the prediction for the point, using the current weights (model)
                pred, vals = self.forward_step(dat)
                # adjust the weights (model) to account for whether we're incorrect
                self.backprop(vals, pred, dat, train_y[j])
                loss += (abs(pred - train_y[j])**2).sum()
        print("Current loss: " + str(loss))
        return loss



    def predict(self, train_x, train_y, labels, verbose=False):
#             loss = 0    # 
#             correct = 0 # Number of trials that were guessed correct            
#             for j in range(0,train_x.shape[0]):
            dat = train_x#[j]
            if self.bias:
                dat = np.append(train_x, [1])#train_x[j]

            # get the prediction for the point, using the current weights (model)
            pred, vals = self.forward_step(dat)
            # adjust the weights (model) to account for whether we're incorrect
            self.backprop(vals, pred, dat, train_y)#train_y[j]
#                 loss += (abs(pred - train_y[j])**2).sum()
#                 compare = pred==train_y[j]
#                 if compare.all():
#                     correct+=1
#             print("Current loss: "+ str(loss))
            #print(f"Correct: {correct} | Total:{train_x.shape[0]}")
            #pred: [.5, .75, .69, .99]
            #labl: [theft, prostitution, GTA, Tax Fraud]
            
            return str(labels[pred.argmax()])