from KNN_FOR_MINST_DATABSE.nearest_neighbor import *
from KNN_FOR_MINST_DATABSE.untils import *


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
