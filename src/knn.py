from sklearn.neighbors import KNeighborsClassifier
from utils import get_total_accuracy, get_waldo_accuracy, build_cv_data
import numpy as np


def knn_algorithm(x_train, y_train, x_test, y_test, k=1):
    # Flatten arrays
    x_train = np.reshape(x_train, (x_train.shape[0], 128 * 128 * 3))
    x_test = np.reshape(x_test, (x_test.shape[0], 128 * 128 * 3))

    clf = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', n_jobs=4)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    # Show accuracy on the total dataset, then only on waldos images
    total_accuracy = get_total_accuracy(y_test, predictions)
    waldo_accuracy = get_waldo_accuracy(y_test, predictions)
    print('Total accuracy of: ' + str(total_accuracy) + '%.')
    print('Waldo accuracy of: ' + str(waldo_accuracy) + '%.')


def knn_cross_validation(x_train, y_train, cross_validation_size, k):
    if cross_validation_size <= 1:
        print('Cross validation size must be 2 at least')
        return False
    for i in range(cross_validation_size):
        print('Cross validation iteration ' + str(i) + ':')
        x_train, y_train, validation_data, validation_label = \
            build_cv_data(x_train, y_train, i, cross_validation_size)

        if x_train is None:
            print("Failed to build cross validation data")
            return False

        knn_algorithm(x_train, y_train, validation_data, validation_label, k)

    return True
