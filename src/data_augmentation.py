from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib.pyplot import imread
import numpy as np
import sys
import os

WALDO_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/'
WALDO_DUPLICATION_NB = 100
NOTWALDO_DUPLICATION_NB = 2


def get_images_data(dir):
    return np.array([np.array(imread(WALDO_DIR + dir + '/waldo/' + fname))
                     for fname in os.listdir(WALDO_DIR + dir + '/waldo')]),\
            np.array([np.array(imread(WALDO_DIR + dir + '/notwaldo/' + fname))
                      for fname in os.listdir(WALDO_DIR + dir + '/notwaldo')])


def data_augmentation(waldos, notWaldos, dir):
    datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode='nearest')

    print('Waldo data augmentation...')
    for waldo in waldos:
        waldo = waldo.reshape((1,) + waldo.shape)

        i = 0
        for batch in datagen.flow(waldo, batch_size=1,
                                  save_to_dir=WALDO_DIR + dir + '/waldoOther', save_prefix='waldo', save_format='jpeg'):
            i += 1
            if i >= WALDO_DUPLICATION_NB - 1:
                break

    print('Not waldo data augmentation...')
    for not_waldo in notWaldos:
        not_waldo = not_waldo.reshape((1,) + not_waldo.shape)

        i = 0
        for batch in datagen.flow(not_waldo, batch_size=1,
                                  save_to_dir=WALDO_DIR + dir + '/notWaldoOther', save_prefix='waldo', save_format='jpeg'):
            i += 1
            if i >= NOTWALDO_DUPLICATION_NB - 1:
                break


if __name__ == '__main__':
    if len(sys.argv) > 1:
        waldos, notWaldos = get_images_data(sys.argv[1])
        data_augmentation(waldos, notWaldos, sys.argv[1])
    else:
        print('Usage: build_csv.py [image directory]')
