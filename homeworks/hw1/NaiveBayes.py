import numpy as np


class NaiveBayse():
    def __init__(self):
        pass


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
            dataset += [line]
    dataset = np.array(dataset)
    return dataset


if __name__ == '__main__':
    dataset = loadDataset("agaricus-lepiota.data")
    print(dataset.shape)
