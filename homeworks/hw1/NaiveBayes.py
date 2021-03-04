import numpy as np
import random
from math import sqrt, pi, exp


class NaiveBayse():

    def __calculateMedians(self, X):
        medians = np.sum(X, axis=0)
        medians = medians/len(X)
        return medians

    def __calculateSigmas(self, X):
        medians = self.__calculateMedians(X)
        sigmas = X - medians
        sigmas = sigmas*sigmas
        sigmas = np.sum(sigmas, axis=0)
        sigmas = sigmas/(len(X)-1)
        return sigmas

    def __calculateGausians(self, data):
        categoricalData = []
        for c in self.classes:
            cData = []
            for i in range(len(data[1])):
                if(data[1][i] == c):
                    cData += [data[0][i]]
            cData = np.array(cData)
            categoricalData += [cData]
        categoricalGausians = []
        for cData in categoricalData:
            sigmas = self.__calculateSigmas(cData)
            medians = self.__calculateMedians(cData)
            gausian = list(zip(medians, sigmas))
            categoricalGausians += [gausian]
        return categoricalGausians

    def __init__(self, data):
        self.classes, counts = np.unique(data[1], return_counts=True)
        self.classProbablities = []
        for i in range(len(self.classes)):
            self.classProbablities += [counts[i]/counts.sum()]
        self.classProbablities = np.array(self.classProbablities)
        self.gausians = self.__calculateGausians(data)

    def __gausian(self, c, f, x):
        (m, sigma) = self.gausians[c][f]
        if(sigma == 0):
            return 1
        a = 1/(sqrt(2*pi*sigma*sigma))
        b = exp((-1*(x-m)*(x-m))/(2*sigma*sigma))
        return a*b

    def __pEvidance(self, x):
        pe = 0
        for c in self.classes:
            p = self.classProbablities[c]
            for i in range(len(x)):
                p = p * self.__gausian(c, i, x[i])
            pe += p
        return pe

    def __probablities(self, x):

        p_e = self.__pEvidance(x)
        p_c = []
        for c in self.classes:
            p = self.classProbablities[c]
            for i in range(len(x)):
                p = p * self.__gausian(c, i, x[i])
            p_c += [p]
        return np.array(p_c)

    def predict(self, X):
        y = []
        for x in X:
            p_c = self.__probablities(x)
            y += [np.argmax(p_c)]
        return np.array(y)


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
            nData[i, j] = cat[j].index(data[i, j])

    return nData


def accuracy_score(a, b):
    c = 0
    for i in range(len(a)):
        if(a[i] == b[i]):
            c += 1
    return (c/len(a))*100


if __name__ == '__main__':
    dataset = loadDataset("agaricus-lepiota.data")
    dataset = convert2Numbers(dataset)
    train, test = testDataSeparator(dataset)
    print(train[0].shape, train[1].shape)
    clf = NaiveBayse(train)
    y_pred = clf.predict(test[0])
    print("Accuracy Score: ", accuracy_score(y_pred, test[1]))
