from KNN_FOR_MINST_DATABSE.nearest_neighbor import *
from KNN_FOR_MINST_DATABSE.untils import *


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



def test_nearest_neighbor_classifier(slices):
    images_matrix, labels_matrix = load_mnist(mnist_path)
    text_images_matrix, text_labels_matrix = load_mnist(mnist_path, kind='t10k')
    NNAPP = NearestNeighbor(images_matrix, labels_matrix)
    ret = NNAPP.predict(text_images_matrix[0:slices])
    # print('predict_labels:', ret)
    # print('real_labels:', text_labels_matrix[0:slices])
    print(np.mean(ret == text_labels_matrix[0:slices]))


def test_k_nearest_neighbor_classifier(k, slices):
    images_matrix, labels_matrix = load_mnist(mnist_path)
    text_images_matrix, text_labels_matrix = load_mnist(mnist_path, kind='t10k')
    NNAPP = KNearestNeighbor(images_matrix, labels_matrix, k)
    ret = NNAPP.predict(text_images_matrix[0:slices])
    # print(np.mean(text_labels_matrix == ret))
    # print('predict_labels:', ret)
    # print('real_labels:', text_labels_matrix[0:slices])
    print(np.mean(ret == text_labels_matrix[0:slices]))


if __name__ == '__main__':
    N = 300
    test_k_nearest_neighbor_classifier(3, N)
    test_k_nearest_neighbor_classifier(5, N)
    test_k_nearest_neighbor_classifier(7, N)
    test_k_nearest_neighbor_classifier(9, N)
    test_nearest_neighbor_classifier(N)
