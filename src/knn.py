from sklearn.neighbors import KNeighborsClassifier
from utils import *


def knn_algorithm(train_data, train_label, test_data, test_label, k=1):
    clf = KNeighborsClassifier(n_neighbors=k, algorithm='ball_tree', n_jobs=4)
    clf.fit(train_data, train_label)
    predictions = clf.predict(test_data)

    # Show accuracy on the total dataset, then only on waldos images
    total_accuracy = get_total_accuracy(test_label, predictions)
    waldo_accuracy = get_waldo_accuracy(test_label, predictions)
    print('Total accuracy of: ' + str(total_accuracy) + '%.')
    print('Waldo accuracy of: ' + str(waldo_accuracy) + '%.')


def knn_cross_validation(train_data, train_label, cross_validation_size, k):
    if cross_validation_size <= 1:
        print('Cross validation size must be 2 at least')
        return False
    for i in range(cross_validation_size):
        print('Cross validation iteration ' + str(i) + ':')
        train_data, train_label, validation_data, validation_label = \
            build_cv_data(train_data, train_label, i, cross_validation_size)

        if train_data is None:
            print("Failed to build cross validation data")
            return False

        knn_algorithm(train_data, train_label, validation_data, validation_label, k)

    return True
