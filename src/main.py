import matplotlib
matplotlib.use('agg')
from knn import knn_algorithm, knn_cross_validation
from svm import svm_algorithm, svm_cross_validation
from cnn import cnn_algorithm
from cnn_finetune import cnn_ft_algorithm
from utils import get_data
import numpy as np

CROSS_VALIDATION_SIZE = 4
DATA_FOLDER = '128'
SEED = 42


if __name__ == '__main__':
    np.random.seed(SEED)

    # x_train, y_train, x_test, y_test = get_data(DATA_FOLDER)

    # Cross validation example to choose K for KNN algorithm
    # for k in range(1, 5):
    #     print('Test with K at ' + str(k) + ':')
    #     knn_cross_validation(x_train, y_train, cross_validation_size=4, k=k)

    # Basic usage of KNN algorithm
    # knn_algorithm(x_train, y_train, x_test, y_test, 5)

    # Cross validation example for SVM algorithm
    # svm_algorithm(x_train, y_train, x_test, y_test)

    # Convolution neural network
    # cnn_algorithm()
    cnn_ft_algorithm()
