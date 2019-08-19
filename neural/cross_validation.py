import numpy as np
import random

class CrossValidation:
    def __init__(self, dataset=[], kpart=4, seed=45):
        self.seed = seed
        self.dataset= dataset
        self.parts = kpart
        self.kFolds = []
        self.folds()

    def folds(self):
        random.Random().shuffle(self.dataset)
        length_fold = round(len(self.dataset) / self.parts)
        folds = [self.dataset[x:x+length_fold] for x in range(0, len(self.dataset), length_fold)]

        kfolds = []
        for i in range(self.parts):
            item = folds[i]
            train, test= [],[]
            for f in folds:
                if  np.array_equal(f, item) == False:
                    for j in f:
                        train.append(j)
                else:
                    for j in item:
                        test.append(j)
                    test.append
            kfolds.append((train, test))
        
        self.kFolds = kfolds

    def getKFolds(self):
        return self.kFolds
