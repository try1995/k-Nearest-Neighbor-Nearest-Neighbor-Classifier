import numpy as np


class NearestNeighbor:
    def __init__(self, X, y):
        self.ytr = y
        self.Xtr = X

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)

        for i in range(num_test):
            # print("正在比较第%s个" % (i+1))
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            min_index = np.argmin(distances)
            Ypred[i] = self.ytr[min_index]

        return Ypred
