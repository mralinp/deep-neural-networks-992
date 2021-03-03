import numpy as np
from tqdm import tqdm


class LogesticRegression:
    '''
        This class is a LogesticReggression Classifier.
        It could use pretrained weights or random weights.
        numClasses : number of output classes
        batchSize  : Batch size for training
        weights    : pretrained weights (None for random selection)
        dim        : input data dimentions
        lr         : learning rate.
    '''

    # Constructor
    def __init__(self, weights=None, numClasses=10, lr=0.5, batchSize=32, dim=(5,), normalizeData=True):
        self.baias = np.random.uniform(low=0, high=+1, size=(1,))
        self.weights = np.random.uniform(low=0, high=+1, size=dim)
        if(weights != None):
            self.weights = weights
        self.lr = lr
        self.batchSize = batchSize
        self.dim = dim
        self.numClasses = numClasses
        self.normalizeData = normalizeData
        print(self.weights)

    # Gets training data and trains the model.
    def fit(self, X, y):
        pass

    # Gets Training data and predicts the output values
    def predict(self, X):
        y = self.weights.transpose()*X
        y = self.__activation(y + self.baias)
        return y

    # Sigmoid activation function
    def __sigmoid(self, x, deriv=False):
        res = 1/1+np.exp(-1*x)
        if (deriv == True):
            res = res*(1-res)
        return res

    # Updating weights at the end of each batch
    def __updateWeights(self, x):
        pass

    # Min squear error (MSE) loss function
    def __loss(self, X, y):
        pass


if __name__ == "__main__":
    dataset = []
    with open("agaricus-lepiota.data", 'r') as file:
        while (True):
            f = file.readline()
            if(not f):
                break
            f = f.split(',')
            f[-1] = f[-1][:1]
            f = list(map(lambda x: ord(x)-ord('a'), f))
            dataset = dataset + [f]
        dataset = np.array(dataset)
    print(dataset.shape)
    X, y = dataset[:, :-1], dataset[:, -1]
    print(X.shape, y.shape)
    lr = LogesticRegression(numClasses=10)
