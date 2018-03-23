import pandas as pd
import numpy as np


def read_csv(csv_path):
    return pd.read_csv(filepath_or_buffer=csv_path)


def organize_data(data_array):
    return data_array[:, :-1], data_array[:, -1]


def timer(start, end):
    hours, rem = divmod(end-start, 3600)
    minutes, seconds = divmod(rem, 60)
    print("Process time: {:0>2}:{:0>2}:{:05.2f}".format(int(hours),int(minutes),seconds))


def get_total_accuracy(label, predictions):
    n = label.shape[0]
    cnt = 0

    for i in range(0, n):
        if label[i] == predictions[i]:
            cnt += 1
    return round(cnt / n * 100, 2)


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


def build_cv_data(train_data, train_label, idx, cross_validation_size):
    final_data = None
    final_label = None
    check = False

    splitted_data = np.array_split(train_data, cross_validation_size)
    splitted_label = np.array_split(train_label, cross_validation_size)

    if idx >= cross_validation_size or idx < 0:
        print("Cross validation index invalid.")
        return None, None, None, None

    # print(splitted_data.shape)
    validation_data = splitted_data[idx]
    validation_label = splitted_label[idx]

    for i in range(cross_validation_size):
        if check is False and i != idx:
            final_data = splitted_data[i]
            final_label = splitted_label[i]
            check = True
        if i != idx:
            final_data = np.concatenate((final_data, splitted_data[i]))
            final_label = np.concatenate((final_label, splitted_label[i]))

    return final_data, final_label, validation_data, validation_label
