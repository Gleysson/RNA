from neural import mlpr as network 
from neural import cross_validation as cv
from neural import search_grid as sg
from neural import dataset as db
from neural import graphic as gp

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as split
import numpy as np

REALIZATIONS = 20
N_SPLITS = 5
database = db.dataset()
x_data, y_data = database.regressionDatabase()
graph = gp.Graph()

# j = nn.test(x, activation="linear")

grid = sg.SearchGrid(type="regression")


RMSE, MSE, WEIGTHS = [], [], []

for i in range(REALIZATIONS):

    x_train, x_test, y_train, y_test = split(x_data, y_data, test_size=0.2)
    nn = network.mlp(inputs=x_train, outputs=y_train, outLayer=1, hiddenLayer=4, eta=0.01)
    nn.trainer(1000, activation="linear")
    
    print(" > -----------------------------------------------")
    print(" > Realizações: ",i+1 )
    print(" > MSE: ", nn.mse , "RMSE:", nn.rmse)
    print(" >")

    predict = nn.test(x_test, outputs=y_test)
    MSE.append(nn.mse)
    RMSE.append(nn.rmse)
    WEIGTHS.append(nn.getWeigths())

BEST_INDEX = np.argmin(MSE)
predict = nn.test(x_data, y_data, weigths=WEIGTHS[BEST_INDEX], activation="linear")
graph.plot(t=x_data, line=y_data, x_test=x_test, y_test=y_test, predict=predict)

print(" > -----------------------------------------------")
print(" > Resultado Final: " )
print(" > Média MSE: ", np.mean(MSE) , " Média RMSE:", np.mean(RMSE))
print(" > Desvio Padrão MSE: ", np.std(MSE) , " Desvio Padrão RMSE:", np.std(RMSE))


