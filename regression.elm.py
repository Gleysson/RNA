from neural import elmr as network 
from neural import cross_validation as cv
from neural import search_grid as sg
from neural import dataset as db
from neural import graphic as gp


from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as split
import numpy as np

REALIZATIONS = 1
N_SPLITS = 5
database = db.dataset()
x_data, y_data = database.regressionDatabaseELM()
graph = gp.Graph()

# j = nn.test(x, activation="linear")
# graph.plot(t=x, line=y, predict= j)


# search grid
grid = sg.SearchGrid(type="regression")

print("## Problema: REGRESSION ")
print("## Treinamento: KFold - K:", N_SPLITS)     
for epoch in grid.getEpochs():
    for eta in grid.getEtas():
        for neuron in grid.getNeurons():

            x_train, x_test, y_train, y_test = split(x_data, y_data, test_size=0.2)
            MSE_FINAL, RMSE_FINAL = [], []

            print("--------------------------------------------------" )
            print(" > Epoch", epoch, "Eta:", eta, "Neurons:", neuron )

            for i in range(REALIZATIONS):
                kf = KFold(n_splits=N_SPLITS, shuffle=True)

                MSE, RMSE = [], []
                WEIGTHS = []
                for train_index, test_index in kf.split(x_train):
                    
                    X_train, X_test = x_train[train_index], x_train[test_index]
                    Y_train, Y_test = y_train[train_index], y_train[test_index]

                    nn = network.ELM(inputs=X_train, outputs=Y_train, outLayer=1, hiddenLayer=neuron,)
                    nn.trainer( activation="linear")
                    
                    predict = nn.test(X_test, Y_test, activation="linear")
                    MSE.append(nn.mse)
                    RMSE.append(nn.rmse)
                    WEIGTHS.append(nn.getWeigths())

                BEST_INDEX = np.argmin(RMSE)
                predicrt = nn.test(x_test, y_test, weigths=WEIGTHS[BEST_INDEX], activation="linear")
                MSE_FINAL.append(nn.mse)
                RMSE_FINAL.append(nn.rmse)

            
        
            print(" > MEAN MSE: ", np.mean(MSE_FINAL) , "MEAN RMSE:", np.mean(RMSE_FINAL))
            print(" >")
            graph.plot(t=x_test, line=y_test, predict=predicrt)




