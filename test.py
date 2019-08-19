#!/usr/bin/python
from sklearn.model_selection import train_test_split
import numpy as np

def sigmoid(x):
    return 1.0/(1.0 + np.exp(-x))

def sigmoid_der(x):
    return x*(1.0 - x)

class NN:
    def __init__(self, inputs):
        self.inputs = inputs
        self.l=len(self.inputs)
        self.outLayer = 1
        self.hiddenLayer = 5
        self.numAttrs=len(self.inputs[0]) # insert bias in input layer 

        self.wi=np.random.random((self.numAttrs, self.hiddenLayer))
        self.wh=np.random.random((self.hiddenLayer, self.outLayer))

    def think(self, inp):
        s1=sigmoid(np.dot(inp, self.wi))
        s2=sigmoid(np.dot(s1, self.wh))
        return s2

    def train(self, inputs,outputs, it):
        for i in range(it):
            
            l0=inputs
            l1=sigmoid(np.dot(l0, self.wi))
            # print('Layer 1:',l1)
            l2=sigmoid(np.dot(l1, self.wh))

            l2_err= outputs - l2

            l2_delta = np.multiply(l2_err, sigmoid_der(l2)) 
            l1_err=np.dot(l2_delta, self.wh.T)
            l1_delta=np.multiply(l1_err, sigmoid_der(l1)) 

            self.wh+=np.dot(l1.T, l2_delta)
            self.wi+=np.dot(l0.T, l1_delta)


x, y = [], []
err = 0.2
for i in range(1):
    x.append([ 1 + np.random.uniform(-err,err), 1 + np.random.uniform(-err,err)]) 
    x.append([ 0 + np.random.uniform(-err,err), 0 + np.random.uniform(-err,err)]) 
    x.append([ 1 + np.random.uniform(-err,err), 0 + np.random.uniform(-err,err)]) 
    x.append([ 0 + np.random.uniform(-err,err), 1 + np.random.uniform(-err,err)]) 
    # dataset = np.concatenate((inputs, outputs), axis=1)
    y.append([0])
    y.append([0])
    y.append([1])
    y.append([1])

    y = np.array(y)
    x = np.array(x)

            
n= NN(x)

print(n.think(x))
n.train(x, y, 1000)
print("Depois")
print(n.think(x))