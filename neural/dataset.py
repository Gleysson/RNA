import numpy as np
import pandas as pd

class dataset:

    def __init__(self):
        self.d = []
        self.x = []

    def frange(self,start, stop, step):
        i = start
        while i <= stop:
            yield i
            i += step


    def irisFlower(self):
        dataset = pd.read_csv('./database/iris/iris.data', sep=',') # leitura do dataset
        flowers = np.array(dataset['Classe']) # converte classe literal para 1 ou 0
        
        for flower in flowers:
            if(flower == 'Iris-setosa'):
                self.d.append([1, 0, 0])
            elif(flower == 'Iris-versicolor'):
                self.d.append([0, 1, 0])
            else:
                self.d.append([0, 0, 1])

        self.x = np.array(dataset.drop(['Classe'], axis=1)) # x - entradas do dataset

        return np.array(self.x), np.array(self.d)


    def xorDatabase(self):
        err = 0.15
        for i in range(50):
            self.x.append([ 1 + np.random.uniform(-err,err), 1 + np.random.uniform(-err,err)]) 
            self.x.append([ 0 + np.random.uniform(-err,err), 0 + np.random.uniform(-err,err)]) 
            self.x.append([ 1 + np.random.uniform(-err,err), 0 + np.random.uniform(-err,err)]) 
            self.x.append([ 0 + np.random.uniform(-err,err), 1 + np.random.uniform(-err,err)]) 

            self.d.append([0])
            self.d.append([0])
            self.d.append([1])
            self.d.append([1])

        return np.array(self.x), np.array(self.d)


    def regressionDatabase(self):
        x = np.arange(0.0, 5, 0.1)
        y = 3 * np.sin(x) + 1 + np.random.random(len(x))
        return np.array(x), np.array(y)
    
    def regressionDatabaseELM(self):
        x = np.arange(0.0, 5, 0.05)
        y = 2 * np.sin(x) + 3 + np.random.random(len(x))
        return np.array(x), np.array(y)

    def breastCancer(self):

        dataset = pd.read_csv('./database/breast-cancer/breast-cancer-wisconsin.data', sep=',') # leitura do dataset
        breast = np.array(dataset['Classe']) # converte classe literal para 1 ou 0
        
        for pattern in breast:
            if(pattern == 4):
                self.d.append([1, 0])
            elif(pattern == 2):
                self.d.append([0, 1])
           

        self.x = np.array(dataset.drop(['Classe', 'id'], axis=1)) # x - entradas do dataset

        return np.array(self.x, dtype=np.float64), np.array(self.d , dtype=np.float64)


    def vertebralColumn(self):
        
            dataset = pd.read_csv('./database/colunm-vertebral/column_3C.dat', sep=' ') # leitura do dataset
            breast = np.array(dataset['Classe']) # converte classe literal para 1 ou 0
            
            for pattern in breast:
                if(pattern == 'DH'):
                    self.d.append([1, 0, 0])
                elif(pattern == 'SL'):
                    self.d.append([0, 1, 0])
                else:
                    self.d.append([0, 0, 1])

            self.x = np.array(dataset.drop(['Classe'], axis=1)) # x - entradas do dataset

            return np.array(self.x, dtype=np.float64), np.array(self.d , dtype=np.float64)

    def dermatology(self):
        
            dataset = pd.read_csv('./database/dermatology/dermatology.data', sep=',') # leitura do dataset
            breast = np.array(dataset['Classe']) # converte classe literal para 1 ou 0
            
            for pattern in breast:
                aux = [0,0,0,0,0,0]
                aux[pattern-1] = 1
                self.d.append(aux)
                

            self.x = np.array(dataset.drop(['Classe'], axis=1)) # x - entradas do dataset

            return np.array(self.x, dtype=np.float64), np.array(self.d , dtype=np.float64)









