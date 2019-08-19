class SearchGrid:

    def __init__(self, type="classifier"):
        self.etas = []
        self.epochs = []
        self.neurons = []
        self.setValues(type)

    def setValues(self, type):

        if(type=='classifier'):
            self.etas = [0.06, 0.08, 0.1 , 0.12]
            self.epochs = [500, 500, 700, 1000, 1500, 2000]
            self.neurons = [2,4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26,28 ,30 ]
        else:
            self.etas = [0.001 , 0.005 , 0.008 , 0.01, 0.05, 0.08, 0.01]
            self.epochs = [200, 500, 700, 1000, 1200, 1500, 2000, 3000]
            self.neurons = [4, 6, 8, 10, 12, 14, 16, 20, 22, 25, 28, 30 ]
        
        self.neuronsElm = [4, 6, 8, 10, 12, 14, 16, 20, 22, 25, 28, 30, 32, 34 ,36, 38, 40, 42, 44, 46, 48, 50 ]


    def getEtas(self):
        return self.etas

    def getEpochs(self):
        return self.epochs

    def getNeurons(self):
        return self.neurons
    def getNeuronsELM(self):
        return self.neuronsElm

