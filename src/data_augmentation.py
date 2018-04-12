from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from matplotlib.pyplot import imread, imshow, show, savefig
import numpy as np
import sys
import os
import cv2

WALDO_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/'
WALDO_DUPLICATION_NB = 100
NOTWALDO_DUPLICATION_NB = 2
NOISE_DUPLICATION = 30


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


def get_waldos_data(path):
    waldos = []
    for fname in os.listdir(path + '/'):
        waldos.append(cv2.imread(path + '/' + fname))
    return np.array(waldos)


def data_add_noise(waldos, path):
    for idx, waldo in enumerate(waldos):
        for i in range(10):
            noise = (0.4 * np.random.randn(*waldo.shape)) * 10
            img = waldo + noise
            cv2.imwrite(path + '/noisy_' + str(idx) + '_' + str(i) + '.jpg', img)


if __name__ == '__main__':
    if len(sys.argv) > 2:
        if sys.argv[1] == 'aug':
            waldos, notWaldos = get_images_data(sys.argv[2])
            data_augmentation(waldos, notWaldos, sys.argv[2])
        elif sys.argv[1] == 'noise':
            waldos = get_waldos_data(sys.argv[2])
            data_add_noise(waldos, sys.argv[2])
    else:
        print('Usage: data_augmentation.py [aug or noise] [image directory]')
