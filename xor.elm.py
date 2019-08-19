from neural import elm as network 
from neural import cross_validation as cv
from neural import search_grid as sg
from neural import dataset as db
from neural import graphic as gp

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as split
import numpy as np



N_SPLITS = 5
REALIZATIONS = 1

# import dataset
database = db.dataset()
x_data, y_data = database.xorDatabase()

graph = gp.Graph()
# search grid
grid = sg.SearchGrid()

print("## Problema: XOR ELM")
print("## Treinamento: KFold - K:", N_SPLITS)     

for neuron in grid.getNeurons():

    x_train, x_test, y_train, y_test = split(x_data, y_data, test_size=0.2)

    TAXAS_FINAL = []


    print("--------------------------------------------------" )
    print(" > Neurons:", neuron )

    for i in range(REALIZATIONS):
        kf = KFold(n_splits=N_SPLITS, shuffle=True)

        count = 0;


        HITS = []
        WEIGTHS = []
        for train_index, test_index in kf.split(x_train):
            
            count += 1
            X_train, X_test = x_train[train_index], x_train[test_index]
            Y_train, Y_test = y_train[train_index], y_train[test_index]
            
            nn = network.ELM(inputs=X_train, outputs=Y_train, outLayer=1, hiddenLayer=neuron)
            nn.trainer()
            predict = nn.test(X_test)
            HITS.append(nn.accuracy(Y_test, predict))
            WEIGTHS.append(nn.getWeigths())

        BEST_INDEX = np.argmax(HITS)
        Y_PREDICT = nn.test(x_test, weigths=WEIGTHS[BEST_INDEX])

    TAXAS_FINAL.append(nn.accuracy(y_test, Y_PREDICT))
    predict = nn.test(x_data, weigths=WEIGTHS[BEST_INDEX])
    graph.plotXor( nn.test , w=WEIGTHS[BEST_INDEX], x_train=x_train,  x_test=x_test, predict=predict)

    print(" > Acurácia: ", np.mean(TAXAS_FINAL))
    print(" >")

                
           