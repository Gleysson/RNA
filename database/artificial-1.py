
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split


class Iris:

    LIMIAR = -1
    NUM_EPOCAS = 300
    NUM_REALIZACOES = 2 
    TAXA_APRENDIZAGEM = 0.005

    RESULT_PESOS_TESTE = []
    RESULT_ACURACIA_TESTE = []
    RESULT_LIMIAR_TESTE = []
    RESULT_CONFUSAO_TESTE = []


    # EMBARALHA DADOS DE TESTE ----------------------------------------
    def embaralhar(self, x, y, percent):
        return train_test_split(x, y, test_size=percent)

    def database(self):
        entradas = []
        saidas = []
        for i in range(40):

            if i < 10:
                d = 0 
                x1 = 0 + np.random.uniform(-0.02, 0.02)
                x2 = 0 + np.random.uniform(-0.02, 0.02)
                
            elif i >= 10 and i < 20:
                d = 0 
                x1 = 1 + np.random.uniform(-0.02, 0.02)
                x2 = 0 + np.random.uniform(-0.02, 0.02)

            elif i >= 20 and i < 30:
                d = 0 
                x1 = 0 + np.random.uniform(-0.02, 0.02)
                x2 = 1 + np.random.uniform(-0.02, 0.02)

            else:
                d = 1 
                x1 = 1 + np.random.uniform(-0.02, 0.02)
                x2 = 1 + np.random.uniform(-0.02, 0.02)

            entradas.append([x1,x2])
            saidas.append(d)

        return entradas, saidas

    # FUNÇÃO DE ATIVAÇÃO DEGRAU ---------------------------------------
    def degrau(self, u):
        if u >= 0:
            return 1
        return 0

    # FUNÇÃO SOMA -----------------------------------------------------
    def soma(self, w, entradas, limiar):
        return np.dot(w,entradas) + (limiar)

    # REGRA DE APRENDIZAGEM -------------------------------------------
    def regraAprendizagem(self, valor, taxa, y_treino, y, x_treino):
        return valor + taxa * (y_treino - y) * x_treino

    # CÁLCULO DE ACURÁCIA ---------------------------------------------
    def acuracia(self, x,y):
        return accuracy_score(x, y)

    # FUNÇÃO DE TESTE -------------------------------------------------
    def teste(self, w, limiar, x_teste, y_teste):
        y_saida = []
        for i in range(len(x_teste)):
            y = self.degrau(self.soma(w, x_teste[i], limiar))
            y_saida.append(y)

        return  accuracy_score(y_teste, y_saida), confusion_matrix(y_teste, y_saida)

    # RETORNA PESOS RANDOM --------------------------------------------
    def random(self, t):
        return np.random.random(t)
    

    # RETORNA RANGE COM PASSO FLOAT -----------------------------------
    def frange(self,start, stop, step):
        i = start
        while i <= stop:
            yield i
            i += step
    # -----------------------------------------------------------------

    def graph(self,x_treino,x_teste,y_treino,y_teste):
        pos = np.argmax(self.RESULT_ACURACIA_TESTE)
        print(pos, "Grafico posição")

        w = self.RESULT_PESOS_TESTE[pos]
        limiar = self.RESULT_LIMIAR_TESTE[pos]

        print("Limiar", limiar)
        print("w", w)

        for i in self.frange(-0.1,1.1,0.05):
            for j in self.frange(-0.1,1.1,0.05):
                x = [i,j]
                if self.degrau(self.soma(w, x, self.LIMIAR)) == 1:
                    plt.plot( i, j, 'go') # green bolinha
                else:
                    plt.plot( i, j, 'ro') # red triangulo

   

        for i in range(len(y_treino)):
            if y_treino[i] == 1:
                plt.plot( x_treino[i][0], x_treino[i][1], 'g*') 
            else:
                plt.plot( x_treino[i][0], x_treino[i][1], 'g^') 

        for i in range(len(y_teste)):
            if y_teste[i] == 1:
                plt.plot( x_teste[i][0], x_teste[i][1], 'b*') 
            else:
                plt.plot( x_teste[i][0], x_teste[i][1], 'b^') 


     
        plt.axis([-0.1, 1.1, -0.1, 1.1])
        plt.title("Superfície de Decisão - Artificial 1")

        plt.grid(True)
        plt.show()

    def treinamento(self,x_treino,y_treino,w):
        count = 0
        while(count < self.NUM_EPOCAS):

            y_saida = []
            for i in range(len(x_treino)):
                y = self.degrau(self.soma(w, x_treino[i], self.LIMIAR))
                y_saida.append(y)

                for j in range(len(w)):
                    w[j] = self.regraAprendizagem(w[j], self.TAXA_APRENDIZAGEM, y_treino[i], y, x_treino[i][j]) 
                    self.LIMIAR = self.regraAprendizagem(self.LIMIAR, self.TAXA_APRENDIZAGEM, y_treino[i], y, x_treino[i][j]) 


            acuracia_treinamento = self.acuracia(y_treino, y_saida)

            count += 1
            if acuracia_treinamento == 1:
                print(y_treino)
                print(y_saida)
                return w 

        return w
    

    def realizar(self):

        # ler dataset e define x -> entradas, d -> saidas desejadas
        x, d = self.database()

        for r in range(self.NUM_REALIZACOES):

            x_treino, x_teste, y_treino, y_teste = self.embaralhar(x, d, 0.15)
            self.LIMIAR = -1

            w = self.random(2) 
            w = self.treinamento(x_treino,y_treino,w)

            acuracia_teste, matriz_confusao_teste = self.teste(w,self.LIMIAR,x_teste,y_teste)
            self.RESULT_PESOS_TESTE.append(w)
            self.RESULT_LIMIAR_TESTE.append(self.LIMIAR)
            self.RESULT_ACURACIA_TESTE.append(acuracia_teste)
            self.RESULT_CONFUSAO_TESTE.append(matriz_confusao_teste)

            print("Realização",r+1," Acurácia DatabaseTest ->", acuracia_teste)
            print(matriz_confusao_teste)
        

        self.graph(x_treino,x_teste,y_treino,y_teste)


       




    
    


   


iris = Iris()
iris.realizar()