import os
import struct
import numpy as np
import matplotlib.pyplot as plt
mnist_path = r"./MINST_DATABASE"


def load_mnist(path, kind='train'):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, '%s-labels.idx1-ubyte' % kind)
    images_path = os.path.join(path, '%s-images.idx3-ubyte' % kind)
    with open(labels_path, 'rb') as lbpath:
        magic, n = struct.unpack('>II', lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, 'rb') as imgpath:
        magic, num, rows, cols = struct.unpack('>IIII', imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)

    return images, labels


def image_show_many(images_matrix, labels_matrix):
    fig, ax = plt.subplots(nrows=2,ncols=5, sharex=True, sharey=True)
    ax = ax.flatten()
    for i in range(10):
        img = images_matrix[labels_matrix == i][0].reshape(28, 28)
        ax[i].imshow(img, cmap='Greys', interpolation='nearest')
        ax[i].set_title(i)
    ax[0].set_xticks([])
    ax[0].set_yticks([])
    plt.tight_layout()
    plt.show()


def image_show(images_matrix):
    fig, ax = plt.subplots()
    img = images_matrix.reshape(28, 28)
    ax.imshow(img, cmap='Greys', interpolation='nearest')
    ax.set_xticks([])
    ax.set_yticks([])
    plt.tight_layout()
    plt.show()


class KNearestNeighbor:
    def __init__(self, X, y, k):
        self.ytr = y
        self.Xtr = X
        self.k = k

    def predict(self, X):
        num_test = X.shape[0]
        Ypred = np.zeros(num_test, dtype=self.ytr.dtype)
        for i in range(num_test):
            # print("正在比较第%s个" % (i+1))
            distances = np.sum(np.abs(self.Xtr - X[i, :]), axis=1)
            k_ls = []
            flag = self.k
            while flag:
                min_index = np.argmin(distances)
                k_ls.append(self.ytr[min_index])
                distances = np.delete(distances, min_index, axis=0)
                flag -= 1
            Ypred[i] = np.argmax(np.bincount(np.array(k_ls)))

        return Ypred


def svm_loss(x, y, w):
    # x一维行向量，y用一个整数表示标签，w权值矩阵
    scores = w.dot(x)
    margins = np.maximum(0, scores - scores[y] + 1)
    margins[y] = 0
    loss_i = np.sum(margins)
    return loss_i


def softmax_loss():
    pass


def full_loss(loss, w, x):
    ret = np.average(loss) + x*np.sum(pow(np.array(w), 2))
    return ret


if __name__ == '__main__':
    images_matrix, labels_matrix = load_mnist(mnist_path)
    # print(type(images_matrix), labels_matrix)
    image_show_many(images_matrix, labels_matrix)