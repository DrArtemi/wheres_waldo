from matplotlib.pyplot import imread
import numpy as np
import pandas as pd
import sys
import os

WALDO_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/'


def get_images_data(dir):
    return np.array([np.array(imread(WALDO_DIR + dir + '/waldo/' + fname)) for fname in os.listdir(WALDO_DIR + dir + '/waldo')]),\
            np.array([np.array(imread(WALDO_DIR + dir + '/notwaldo/' + fname)) for fname in os.listdir(WALDO_DIR + dir + '/notwaldo')])


def build_dataframe(dataArray, label, labelValue):
    data = []

    for img in dataArray:
        data.append(img.flatten('F'))

    df = pd.DataFrame(data)
    df[label] = labelValue

    return df


def build_data(waldos, notwaldos, dir):
    df1 = build_dataframe(waldos, 'waldo', 1)
    df2 = build_dataframe(notwaldos, 'waldo', 0)

    frames = [df1, df2]
    allWaldos = pd.concat(frames)
    allWaldos.to_csv(WALDO_DIR + dir + '/allWaldos_' + dir + '.csv', index=False)


if __name__ == '__main__':
    if len(sys.argv) > 1:
        waldos, notWaldos = get_images_data(sys.argv[1])
        build_data(waldos, notWaldos, sys.argv[1])
    else:
        print('Usage: build_csv.py [image directory]')
