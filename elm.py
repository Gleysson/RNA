from neural import elm as network 
from neural import cross_validation as cv
from neural import search_grid as sg
from neural import dataset as db

from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split as split
import numpy as np



N_SPLITS = 5
REALIZATIONS = 20

# import dataset
database = db.dataset()
x_data, y_data = database.xorDatabase()

x_train, x_test, y_train, y_test = split(x_data, y_data, test_size=0.2)

nn = network.ELM(inputs=x_data, outputs=y_data, outLayer=1, hiddenLayer=12)
predict = nn.trainer()
print(" Acurácia: ",nn.accuracy(y_data, predict))