import numpy as np


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
        return sigmas

    def __init__(self, data):
        self.medians = self.__calculateMedians(data)
        self.sigmas = self.__calculateSigmas(data)
        self.numClasses, counts = np.unique(data[:, 0], return_counts=True)
        self.classProbablities = []
        for i in range(len(self.numClasses)):
            self.probablities += [counts[i]/counts.sum()]
        self.classProbablities = np.array(self.classProbablities)

    def __gausian(self, i, x):
        a = 1/(np.sqrt(2*np.pi*self.sigmas(i)*self.sigmas(i)))
        b = np.exp(
            (-1*(x-self.medians[i])*(x-self.medians[i]))/(2*self.sigmas[i]*self.sigmas[i]))
        return a*b

    def __pEvidance(self, x):

    def __pConditional(self, x):
        for

    def predict(self, X):
        y = []
        for i in X:
            pEvidance = self.__pEvidance(i)
            conditionals = self.__pConditionals(i)
            p = conditionals/pEvidance
            y += [np.argmax(p)]
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
            line = list(map(lambda x: ord(x) - ord('a'), line))
            line[0] = 0 if(line[0] == ord('p') - ord('a')) else 1
            dataset += [line]
    dataset = np.array(dataset)
    return dataset


if __name__ == '__main__':
    dataset = loadDataset("agaricus-lepiota.data")
    X, y = dataset[:, 1:], dataset[:, 0]
    print(X.shape, y.shape)
    print(X[0], "\t", y[0])
    clf = NaiveBayse(X, y)
