import numpy as np
from openCIFAR import unpickle
"""
    Parameters
    Upper-case letters denotes a matrix
    Lower-case letters denotes a vector (or N x 1 matrix)
    The size and the shape of each arguments or parameters are given as comments
"""


class KNN:
    def __init__(self):
        pass

    def train(self, X, y):
        # X is a matrix with N x D
        # y is a vector with N
        self.Xtr = X
        self.ytr = y

    def predict(self, X):
        # X is a matrix with N x D where each row is an example data
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred


if __name__ == "__main__":
    print("KNN Test")
    testD = unpickle("../cifar-10-batches-py/test_batch")
    tX = testD['data']
    ty = testD['labels']
    tX = tX.reshape(10000, 3072)
    ty = np.array(ty)

    model = KNN()
    for i in range(1, 6):
        print("Data Set -", i)
        trainD = unpickle("../cifar-10-batches-py/data_batch_{}".format(i))
        X = trainD['data']
        y = trainD['labels']
        X = X.reshape(10000, 3072)
        y = np.array(y)
        cnt = 0
        model.train(X, y)
        result = model.predict(tX)
        for j in range(tX.shape[0]):
            if ty[j] == result[j]:
                cnt += 1
        print("Matching with K = 1:", cnt / ty.shape[0])
