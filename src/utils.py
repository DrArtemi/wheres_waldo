from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
import pandas as pd
import numpy as np
import pathlib
import shutil
import os


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


def build_dataframe(data_array, label, label_value):
    data = []

    for img in data_array:
        data.append(img.flatten('F'))

    df = pd.DataFrame(data)
    df[label] = label_value

    return df


def load_data(path):
    all_waldos = []
    labels = []

    print('Loading data...')
    for fname in os.listdir(path + 'waldo'):
        all_waldos.append(imread(path + 'waldo/' + fname))
        labels.append(1)

    for fname in os.listdir(path + 'notwaldo'):
        all_waldos.append(imread(path + 'notwaldo/' + fname))
        labels.append(0)

    return np.array(all_waldos), np.array(labels)


def build_train_validation_data(path):
    nb_files = len([1 for x in list(os.scandir(path)) if x.is_file()])
    cnt = 0

    pathlib.Path(path + 'train').mkdir(exist_ok=True)
    pathlib.Path(path + 'train/waldo').mkdir(exist_ok=True)
    pathlib.Path(path + 'train/notwaldo').mkdir(exist_ok=True)
    pathlib.Path(path + 'validation').mkdir(exist_ok=True)
    pathlib.Path(path + 'validation/waldo').mkdir(exist_ok=True)
    pathlib.Path(path + 'validation/notwaldo').mkdir(exist_ok=True)

    nb_test = nb_files / 3

    for fname in os.listdir(path + 'waldo/'):
        shutil.move(path + 'waldo/' + fname, path + 'validation/waldo/' + fname)
        cnt += 1
        if cnt >= nb_test:
            break
    for fname in os.listdir(path + 'waldo/'):
        shutil.move(path + 'waldo/' + fname, path + 'train/waldo/' + fname)

    for fname in os.listdir(path + 'notwaldo/'):
        shutil.move(path + 'notwaldo/' + fname, path + 'validation/notwaldo/' + fname)
        cnt += 1
        if cnt >= nb_test:
            break
    for fname in os.listdir(path + 'notwaldo/'):
        shutil.move(path + 'notwaldo/' + fname, path + 'train/notwaldo/' + fname)


def undo_train_validation_data(path):
    for fname in os.listdir(path + 'train/waldo/'):
        shutil.move(path + 'train/waldo/' + fname, path + 'waldo/' + fname)
    for fname in os.listdir(path + 'validation/waldo/'):
        shutil.move(path + 'validation/waldo/' + fname, path + 'waldo/' + fname)

    for fname in os.listdir(path + 'train/notwaldo/'):
        shutil.move(path + 'train/notwaldo/' + fname, path + 'notwaldo/' + fname)
    for fname in os.listdir(path + 'validation/notwaldo/'):
        shutil.move(path + 'validation/notwaldo/' + fname, path + 'notwaldo/' + fname)

    os.rmdir(path + 'train/waldo')
    os.rmdir(path + 'train/notwaldo')
    os.rmdir(path + 'train')
    os.rmdir(path + 'validation/waldo')
    os.rmdir(path + 'validation/notwaldo')
    os.rmdir(path + 'validation')
