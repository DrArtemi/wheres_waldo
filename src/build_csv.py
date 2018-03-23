from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imread
import numpy as np
import pandas as pd
import sys
import os

WALDO_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/'


def get_images_data(dir):
    return np.array([np.array(imread(WALDO_DIR + dir + '/waldo/' + fname))
                     for fname in os.listdir(WALDO_DIR + dir + '/waldo')]),\
            np.array([np.array(imread(WALDO_DIR + dir + '/notwaldo/' + fname))
                      for fname in os.listdir(WALDO_DIR + dir + '/notwaldo')])


def build_dataframe(data_array, label, label_value):
    data = []

    for img in data_array:
        data.append(img.flatten('F'))

    df = pd.DataFrame(data)
    df[label] = label_value

    return df


def build_data(waldos, notwaldos, dir):
    df1 = build_dataframe(waldos, 'waldo', 1)
    df2 = build_dataframe(notwaldos, 'waldo', 0)

    frames = [df1, df2]
    all_waldos = pd.concat(frames)

    train_waldos, test_waldos = train_test_split(all_waldos, test_size=0.20, random_state=42)

    print("Building train data CSV...")
    train_waldos.to_csv(WALDO_DIR + dir + '/trainWaldos_' + dir + '.csv', index=False)
    print("Building test data CSV...")
    test_waldos.to_csv(WALDO_DIR + dir + '/testWaldos_' + dir + '.csv', index=False)


if __name__ == '__main__':
    np.set_printoptions(threshold=np.nan)
    if len(sys.argv) > 1:
        waldos, notWaldos = get_images_data(sys.argv[1])
        build_data(waldos, notWaldos, sys.argv[1])
    else:
        print('Usage: build_csv.py [image directory]')
