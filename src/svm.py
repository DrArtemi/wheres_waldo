from utils import *
from sklearn import svm


def svm_algorithm(train_data, train_label, test_data, test_label):
    clf = svm.SVC(kernel='linear', verbose=True)
    clf.fit(train_data, train_label)
    predictions = clf.predict(test_data)

    total_accuracy = get_total_accuracy(test_label, predictions)
    waldo_accuracy = get_waldo_accuracy(test_label, predictions)
    print('Total accuracy of: ' + str(total_accuracy) + '%.')
    print('Waldo accuracy of: ' + str(waldo_accuracy) + '%.')


def svm_cross_validation(train_data, train_label, cross_validation_size):
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

        svm_algorithm(train_data, train_label, validation_data, validation_label)

    return True
