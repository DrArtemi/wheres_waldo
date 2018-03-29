from utils import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from itertools import product
from functools import partial
import keras.backend as K
import matplotlib.pyplot as plt
import time
import os

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/128/'
WEIGHTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../weights/'


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"], 'r-x', label="Train Loss")
    ax.plot(history.history["val_loss"], 'b-x', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"], 'r-x', label="Train Accuracy")
    ax.plot(history.history["val_acc"], 'b-x', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)

    plt.show()


def w_categorical_crossentropy(y_true, y_pred, weights):
    nb_cl = len(weights)
    final_mask = K.zeros_like(y_pred[:, 0])
    y_pred_max = K.max(y_pred, axis=1)
    y_pred_max = K.reshape(y_pred_max, (K.shape(y_pred)[0], 1))
    y_pred_max_mat = K.equal(y_pred, y_pred_max)
    for c_p, c_t in product(range(nb_cl), range(nb_cl)):
        final_mask += (weights[c_t, c_p] * y_pred_max_mat[:, c_p] * y_true[:, c_t])
    return K.categorical_crossentropy(y_pred, y_true) * final_mask


def cnn_algorithm():
    # x_train, y_train = load_data(DATA_DIR + 'train/')
    x_test, y_test = load_data(DATA_DIR + 'test/')

    # Weighted crossentropy
    # w_array = np.ones((2, 2))
    # w_array[0, 1] = 1.2
    # w_array[1, 0] = 1.2
    #
    # ncce = partial(w_categorical_crossentropy, weights=w_array)

    undo_train_validation_data(DATA_DIR + 'train/')
    build_train_validation_data(DATA_DIR + 'train/')

    # print(x_train.shape, y_train.shape)

    # Build model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # this converts our 3D feature maps to 1D feature vectors
    model.add(Dense(64))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation('softmax'))

    # Model summary
    model.summary()

    # Build learning process
    model.compile(loss='binary_crossentropy',
                  optimizer='rmsprop',
                  metrics=['accuracy'])

    batch_size = 128

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode='nearest')

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR + 'train/train',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
        DATA_DIR + 'train/validation',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary')

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=500,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        # sample_weight=np.array([1, 100]),
        verbose=1)
    # history = model.fit(x_train, y_train,
    #                     batch_size=batch_size,
    #                     validation_split=0.2,
    #                     verbose=1,
    #                     epochs=10)

    undo_train_validation_data(DATA_DIR + 'train/')
    model.save_weights(WEIGHTS_DIR + 'weights_' + time.strftime("%d-%m-%Y") + '_' + time.strftime('%H:%M') + '.h5')

    # Directly evaluate model
    # score = model.evaluate(x_test, y_test, verbose=0)
    # print('Test loss:', score[0])
    # print('Test accuracy:', score[1])

    # Get predictions ndarray
    predictions = model.predict(x_test, verbose=0)
    print(predictions)
    total_accuracy = get_total_accuracy(y_test, predictions)
    waldo_accuracy = get_waldo_accuracy(y_test, predictions)
    print('Total accuracy of: ' + str(total_accuracy) + '%.')
    print('Waldo accuracy of: ' + str(waldo_accuracy) + '%.')
    plot_loss_accuracy(history)
