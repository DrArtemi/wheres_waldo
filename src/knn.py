from multiprocessing import Process, Queue
from utils import *


class KNearestNeighbor:
    x = None
    y = None
    nb_workers = 4

    def __init__(self):
        pass

    def train(self, data, label):
        self.x = data
        self.y = label

    def predict_data(self, data, q, idx):
        test_size = data.shape[0]
        predictions = np.zeros(test_size)

        for i in range(0, test_size):
            distances = np.sum(np.abs(self.x - data[i, :]), axis=1)
            min_idx = np.argmin(distances)
            predictions[i] = self.y[min_idx]

        q.put((idx, predictions))

    def predict(self, data):
        q = Queue()
        splited_data = np.array_split(data, self.nb_workers)
        predictions = [self.nb_workers]

        workers = [Process(target=self.predict_data,
                           args=(dataPart, q, idx)) for idx, dataPart in enumerate(splited_data)]

        for p in workers:
            p.start()
        for p in workers:
            p.join()
        while not q.empty():
            predictions.append(q.get())

        return self.clean_predictions(predictions)

    @staticmethod
    def clean_predictions(predictions):
        sorted_predict = []
        cnt = 0
        i = 1

        while i <= predictions[0]:
            if predictions[i][0] == cnt:
                sorted_predict.append(predictions[i][1])
                cnt += 1
                i = 0
            i += 1

        final_predict = None
        for item in sorted_predict:
            if final_predict is None:
                final_predict = item
            else:
                final_predict = np.concatenate((final_predict, item), axis=0)

        return final_predict

    @staticmethod
    def get_total_accuracy(label, predictions):
        n = label.shape[0]
        cnt = 0

        for i in range(0, n):
            if label[i] == predictions[i]:
                cnt += 1
        return round(cnt / n * 100, 2)

    @staticmethod
    def get_waldo_accuracy(label, predictions):
        n = label.shape[0]
        cnt = 0
        cnt_waldo = 0

        for i in range(0, n):
            if label[i] == 1:
                cnt_waldo += 1
            if label[i] == 1 and label[i] == predictions[i]:
                cnt += 1
        return round(cnt / cnt_waldo * 100, 2)


def knn_algorithm(train_data, train_label, test_data, test_label):
    knn = KNearestNeighbor()

    knn.train(train_data, train_label)

    predictions = knn.predict(test_data)

    total_accuracy = knn.get_total_accuracy(test_label, predictions)
    waldo_accuracy = knn.get_waldo_accuracy(test_label, predictions)

    print('Total accuracy of: ' + str(total_accuracy) + '%.')
    print('Waldo accuracy of: ' + str(waldo_accuracy) + '%.')


def knn_cross_validation(train_data, train_label, cross_validation_size):
    for i in range(cross_validation_size):
        print('Cross validation iteration ' + str(i) + ':')
        train_data, train_label, validation_data, validation_label = \
            build_cv_data(train_data, train_label, i, cross_validation_size)

        if train_data is None:
            print("Failed to build cross validation data")
            return False

        knn_algorithm(train_data, train_label, validation_data, validation_label)

    return True
