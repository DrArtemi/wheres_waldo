from knn import *
from svm import *
from cnn import *
from utils import load_data
import time
import os

CROSS_VALIDATION_SIZE = 4
SEED = 42


if __name__ == '__main__':
    np.random.seed(SEED)
    # Uncomment below line to see full arrays on print
    # np.set_printoptions(threshold=np.nan)

    # trainWaldos = read_csv(TRAIN_WALDO_CSV_64_RGB).values
    # testWaldos = read_csv(TEST_WALDO_CSV_64_RGB).values
    #
    # data_train, label_train = organize_data(trainWaldos)
    # data_test, label_test = organize_data(testWaldos)

    # Cross validation example to choose K for KNN algorithm
    # print('KNN algorithm cross validation example:')
    # start = time.time()
    # for k in range(1, 5):
    #     print('Test with K at ' + str(k) + ':')
    # knn_cross_validation(data_train, label_train, cross_validation_size=4, k=1)
    # timer(start, time.time())

    # Basic usage of KNN algorithm
    # print('KNN algorithm:')
    # start = time.time()
    # knn_algorithm(data_train, label_train, data_test, label_test, 5)
    # timer(start, time.time())

    # Cross validation example for SVM algorithm
    # print('SVM algorithm cross validation example:')
    # start = time.time()
    # svm_cross_validation(data_train, label_train, cross_validation_size=4)
    # timer(start, time.time())

    # Basic usage of svm algorithm
    # print('SVM algorithm:')
    # start = time.time()
    # svm_algorithm(data_train, label_train, data_test, label_test)
    # timer(start, time.time())

    # Convolution neural network
    print('CNN algorithm:')
    cnn_algorithm()
    # x_test, y_test = load_data(os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/128/' + 'test/')