import numpy as np
from math import exp, sqrt, pi, log
import random
from tqdm import tqdm


class LogesticRegression:
    '''
        This class is a LogisticReggression Classifier.
        It could use pretrained weights or random weights.
        numClasses : number of output classes
        weights    : pretrained weights (None for random selection)
        dim        : input data dimentions
        lr         : learning rate.
    '''

    # Constructor
    def __init__(self, weights=None, numClasses=2, lr=0.1, batchSize=32, epochs=5000, dim=(22,)):
        self.baias = np.random.uniform(low=0.00001, high=+1, size=(1,))
        self.weights = np.random.uniform(low=0.00001, high=+1, size=dim)
        if(weights != None):
            self.weights = weights
        self.lr = lr
        self.batchSize = batchSize
        self.dim = dim
        self.numClasses = numClasses
        self.epochs = epochs

    # Sigmoid activation function
    def sigmoid(self, x):
        res = 1/(1+np.exp(-x))
        return res

    # neg log likelihood function
    def negLoglikelihood(self, y, yPred):
        return -((y * np.log(yPred)) + ((1 - y) * np.log(1 - yPred)))

    # loss function
    def loss(self, y, yPred):
        n = len(y)
        l = self.negLoglikelihood(y, yPred)
        l = l.sum()/n
        return l

    # Gets Training data and predicts the output values
    def predict(self, X):
        y = self.weights*X
        y = np.sum(y, axis=1)
        y = self.sigmoid(y + self.baias)
        return y

    # Caculates the gradiants
    def gradiants(self, x, y):
        error = self.predict(x) - y
        gradient = np.zeros(self.dim)
        for i in range(len(error)):
            for j in range(len(gradient)):
                gradient[j] += error[i] * x[i][j]
        gradient = gradient/len(y)
        gradientOverBaias = np.sum(error)/len(y)
        return gradient, gradientOverBaias

    # Updating weights at the end of each batch
    def __updateWeights(self, gradients, gradientOverBaias):
        self.weights = self.weights - self.lr*gradients
        self.bias = self.baias - self.lr*gradientOverBaias

    # Gets training data and trains the model.
    def fit(self, X, y):
        for i in tqdm(range(self.epochs)):
            yPred = self.predict(X)
            print("Epoch ", i, " loss: ", self.loss(y, yPred))
            g_w, g_b = self.gradiants(X, y)
            self.__updateWeights(g_w, g_b)


def loadDataset(path):
    dataset = []
    with open(path) as file:
        while(True):
            line = file.readline()
            if(not line):
                break
            line = line.split(',')
            line[-1] = line[-1][:-1]
            dataset += [line]
    dataset = np.array(dataset)
    return dataset


def testDataSeparator(data):
    indexes = random.sample(range(len(dataset)), 1000)

    testData = []
    for i in indexes:
        testData += [data[i]]
    testData = np.array(testData)

    trainData = []
    for i in range(len(data)):
        if(i in indexes):
            continue
        trainData += [data[i]]
    trainData = np.array(trainData)
    # (xTrain, yTrain), (xTest, yTest)
    return (trainData[:, 1:], np.array(trainData[:, 0], np.uint8)), (testData[:, 1:], np.array(testData[:, 0], np.uint8))


def convert2Numbers(data):
    nData = np.zeros(data.shape)
    cat = []

    for i in range(data.shape[1]):
        cat += [list(np.unique(data[:, i]))]

    for i in range(len(data)):
        for j in range(len(data[i])):
            if(j != 0):
                nData[i, j] = cat[j].index(data[i, j])/len(cat[j])
            else:
                nData[i, j] = cat[j].index(data[i, j])

    return nData


def accuracy_score(a, b):
    c = 0
    for i in range(len(a)):
        if(a[i] == b[i]):
            c += 1
    return (c/len(a))*100


if __name__ == "__main__":
    dataset = loadDataset("agaricus-lepiota.data")
    dataset = convert2Numbers(dataset)
    (xTrain, yTrain), (xTest, yTest) = testDataSeparator(dataset)
    print(xTrain.shape, xTest.shape)
    print(xTrain[0], yTrain[0])
    clf = LogesticRegression()
    clf.fit(xTrain, yTrain)
    yPred = clf.predict(xTest)
    for i in range(len(yPred)):
        yPred[i] = 1 if(yPred[i] >= 0.5) else 0
    print("accuracy score is: ", accuracy_score(yTest, yPred))
