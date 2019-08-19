from neural import mlp as network 
from neural import cross_validation as cv
from neural import dataset as db
from neural import search_grid as sg

import numpy as np
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as split

N_SPLITS = 5
REALIZATIONS = 20

# import dataset
database = db.dataset()
x_data, y_data = database.vertebralColumn()
# search grid
grid = sg.SearchGrid()

print("## Problema: VERTEBRAL COLUMN ")
print("## Treinamento: KFold - K:", N_SPLITS)     
for epoch in grid.getEpochs():
    for eta in grid.getEtas():
        for neuron in grid.getNeurons():

            x_train, x_test, y_train, y_test = split(x_data, y_data, test_size=0.2)
    
            TAXAS_FINAL = []

            print("---------------------------------------------------" )
            print(" > Epoch", epoch, "Eta:", eta, "Neurons:", neuron )

            for i in range(REALIZATIONS):
                kf = KFold(n_splits=N_SPLITS, shuffle=True)

                count = 0;
            

                HITS = []
                WEIGTHS = []
                for train_index, test_index in kf.split(x_train):
                    
                    count += 1
                    X_train, X_test = x_train[train_index], x_train[test_index]
                    Y_train, Y_test = y_train[train_index], y_train[test_index]
                    
                    nn = network.mlp(inputs=X_train, outputs=Y_train, outLayer=3, hiddenLayer=neuron, eta=eta)
                    nn.trainer(epoch)
                    predict = nn.test(X_test)
                    HITS.append(nn.accuracy(Y_test, predict))
                    WEIGTHS.append(nn.getWeigths())

                BEST_INDEX = np.argmax(HITS)
                Y_PREDICT = nn.test(x_test, weigths=WEIGTHS[BEST_INDEX])
                TAXAS_FINAL.append(nn.accuracy(y_test, Y_PREDICT))
            
        
            print(" > Acurácia: ", np.mean(TAXAS_FINAL), ' Desvio Padrão: ', np.std(TAXAS_FINAL))
            print(" >")

                
                
                


