import numpy as np
import matplotlib.pyplot as plt


class Graph:

    def plot(self, t=[], line=[],x_test=[], y_test=[], predict=[], label="teste"):
            
        fig, ax = plt.subplots()

        for i in range(len(predict)):
            plt.plot( t[i], predict[i], '^',c='orange')
            plt.plot( t[i], line[i], 'o',c='blue')

        for i in range(len(x_test)):
            plt.plot( x_test[i], y_test[i], 'o',c='green')


        ax.set(xlabel='time (s)', ylabel='voltage (mV)',
            title='Regression Problem - MLP')
        ax.grid()
        fig.savefig("./img/"+label+".test.png")

    def frange(self,start, stop, step):
        i = start
        while i <= stop:
            yield i
            i += step

    def plotXor(self,func, w=[], x_train=[],x_test=[], predict=[], label="teste"):
            
        fig, ax = plt.subplots()

      
        data = []
        for i in self.frange(-0.25,1.25,0.05):
             for j in self.frange(-0.25,1.25,0.1):
                 data.append([i, j])
        
        pre = func( data, weigths=w)

        for i in range(len(data)):
            if(pre[i][0] > 0.6):
                plt.plot( data[i][0], data[i][1], 'x',c='green')
            else:
                plt.plot( data[i][0], data[i][1], '.',c='red')
        
        for i in range(len(x_train)):
            plt.plot( x_train[i][0], x_train[i][1], '^',c='orange')

        for i in range(len(x_test)):
            plt.plot( x_test[i][0], x_test[i][1], 'o',c='green')


        ax.set(xlabel='x', ylabel='y',
            title='XOR Problem - MLP')
        ax.grid()
        fig.savefig("./img/xor.test.png")
