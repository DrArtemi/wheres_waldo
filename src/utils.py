from matplotlib.pyplot import imread
from sklearn.utils import shuffle
import numpy as np
import pathlib
import shutil
import time
import os

WALDO_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/'
WEIGHTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../weights/'


def load_data(path):
    all_waldos = []
    labels = []

    print('Loading data...')
    for fname in os.listdir(path + '/waldo'):
        all_waldos.append(imread(path + '/waldo/' + fname))
        labels.append(1)

    for fname in os.listdir(path + '/notwaldo'):
        all_waldos.append(imread(path + '/notwaldo/' + fname))
        labels.append(0)

    return shuffle(np.array(all_waldos), np.array(labels), random_state=42)


def get_data(dir):
    x_train, y_train = load_data(WALDO_DIR + dir + '/train')
    x_test, y_test = load_data(WALDO_DIR + dir + '/test')
    return x_train, y_train, x_test, y_test


def build_cv_data(x_train, y_train, idx, cross_validation_size):
    final_data = None
    final_label = None
    check = False

    splitted_data = np.array_split(x_train, cross_validation_size)
    splitted_label = np.array_split(y_train, cross_validation_size)

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


def build_train_validation_data(path):
    nb_files_w = len([1 for x in list(os.scandir(path + 'waldo/')) if x.is_file()])
    nb_files_nw = len([1 for x in list(os.scandir(path + 'notwaldo/')) if x.is_file()])

    pathlib.Path(path + 'train').mkdir(exist_ok=True)
    pathlib.Path(path + 'train/waldo').mkdir(exist_ok=True)
    pathlib.Path(path + 'train/notwaldo').mkdir(exist_ok=True)
    pathlib.Path(path + 'validation').mkdir(exist_ok=True)
    pathlib.Path(path + 'validation/waldo').mkdir(exist_ok=True)
    pathlib.Path(path + 'validation/notwaldo').mkdir(exist_ok=True)

    nb_test_w = round(nb_files_w / 5)
    nb_test_nw = round(nb_files_nw / 5)

    cnt = 0
    for fname in os.listdir(path + 'waldo/'):
        shutil.move(path + 'waldo/' + fname, path + 'validation/waldo/' + fname)
        cnt += 1
        if cnt >= nb_test_w:
            break
    for fname in os.listdir(path + 'waldo/'):
        shutil.move(path + 'waldo/' + fname, path + 'train/waldo/' + fname)

    cnt = 0
    for fname in os.listdir(path + 'notwaldo/'):
        shutil.move(path + 'notwaldo/' + fname, path + 'validation/notwaldo/' + fname)
        cnt += 1
        if cnt >= nb_test_nw:
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

def save_weights(model):
    model.save_weights(WEIGHTS_DIR + 'weights_' + time.strftime("%d-%m-%Y") + '_' + time.strftime('%H:%M') + '.h5')
