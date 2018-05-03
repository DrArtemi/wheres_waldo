from utils import *
from sklearn import svm


def svm_algorithm(x_train, y_train, x_test, y_test):
    # Flatten arrays
    x_train = np.reshape(x_train, (x_train.shape[0], 128 * 128 * 3))
    x_test = np.reshape(x_test, (x_test.shape[0], 128 * 128 * 3))

    clf = svm.SVC(kernel='linear', verbose=True)
    clf.fit(x_train, y_train)
    predictions = clf.predict(x_test)

    total_accuracy = get_total_accuracy(y_test, predictions)
    waldo_accuracy = get_waldo_accuracy(y_test, predictions)
    print('Total accuracy of: ' + str(total_accuracy) + '%.')
    print('Waldo accuracy of: ' + str(waldo_accuracy) + '%.')


def svm_cross_validation(x_train, y_train, cross_validation_size):
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

        svm_algorithm(x_train, y_train, validation_data, validation_label)

    return True
