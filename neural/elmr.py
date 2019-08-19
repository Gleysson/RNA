#!/usr/bin/python
import numpy as np
import random

class ELM:

    def __init__(self, inputs= [], outputs= [], outLayer = 1, hiddenLayer= 1 ):

        self.x0 = 1
        self.inputs= self.insertBias(inputs) #-> matriz de entrada / padrões 
        self.outputs= self.getOutput(outputs)
        self.hiddenLayer= hiddenLayer #-> quantidade neurônios camada oculta
        self.outLayer= outLayer #-> quantidade neurônios camada de saida
        self.attrs= len(self.inputs[0]) #-> quantidade de variávies/atributos
        self.rmse = 0;
        self.mse = 0;

        self.initWeigths()

    def getOutput(self, out):
        if len(np.array(out).shape) == 1 :
            aux = []
            for i in out:
                aux.append([i])
            return np.array(aux)
        else:
            return np.array(out)

    def initWeigths(self):
        # -> inicializa os pesos de forma randomica
        self.w = np.random.random((self.attrs, self.hiddenLayer))
        self.m = np.random.random((self.hiddenLayer, self.outLayer))

    def insertBias(self, inputs):
        if len(np.array(inputs).shape) == 1 :
            inputs = inputs/np.amax(inputs, axis=0) #maximum of X array

            aux = []
            for i in inputs:
                aux.append([i, self.x0])
            return np.array(aux)
        else:
            return np.insert(inputs, 0, self.x0, axis=1)

    def activation(self, x, act="sigmoid"):
        if(act == "sigmoid"):
            return 1.0/(1.0 + np.exp(-x))
        else:
            return x

    def activation_der(self, x, act="sigmoid"):
        if(act == "sigmoid"):
            return x*(1.0 - x)
        else:
            return 1

    def predict(self, inputs, weigths, act="sigmoid"):
        return self.activation(np.dot(inputs, weigths), act=act)

    def getWeigths(self):
        return self.w, self.m

    def shuffle(self, a, b):
        c = list(zip(a, b))
        random.shuffle(c)
        a, b = zip(*c)
        return np.array(a), np.array(b)

    def test(self, inputs,outputs, weigths=0, activation="sigmoid" ):
        inputs = self.insertBias(inputs)
        if(weigths == 0):
            h = self.predict(inputs, self.w)
            y = self.predict(h, self.m, act=activation)
        else:
            h = self.predict(inputs, weigths[0])
            y = self.predict(h, weigths[1], act=activation)


        if(activation == 'linear'):
            err =  np.sum((outputs - y) **2)
            self.mse = err/len(y)
            self.rmse = np.sqrt(self.mse)

        return y

    def step(self, outputs):
        if(len(outputs) == 1):
            return np.where(outputs > 0.5, 1 , 0)
        else: 
            aux = np.zeros( len(outputs), dtype=int )
            i = np.argmax(outputs)
            aux[i] = 1
            return aux

    def accuracy(self, outputs, predict): 
        hits= 0 
        for i in range(len(predict)):
            if( np.array_equal(outputs[i],self.step(predict[i])) ):
                hits+=1
        return hits / len(predict) 

    def getWeigths(self):
        return self.w, self.m

    def trainer(self, activation='sigmoid'):
            
        x, d = self.inputs, self.outputs
        # x, d = self.shuffle(self.inputs, self.outputs)

        h=self.predict(x, self.w)
        
        self.m = np.linalg.pinv(np.dot(h.T, h)).dot(np.dot(h.T, d))

        y=self.predict(h, self.m, act=activation)

        y_err= d - y
        
        if(activation == 'linear'):
            err = np.sum((y_err) ** 2)
            self.mse = err/len(y_err)
            self.rmse = np.sqrt(self.mse)
        
        return self.activation(h.dot(self.m), act=activation)




