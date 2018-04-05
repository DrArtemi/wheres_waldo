from utils import *
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, Activation, MaxPooling2D, Flatten, Dense, Dropout
from keras.utils import to_categorical
from sklearn.utils import shuffle
from keras import optimizers
import matplotlib.pyplot as plt
import time
import os
import cv2

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


def cnn_algorithm():
    # x_train, y_train = load_data(DATA_DIR + 'train/')
    x_test, y_test = load_data(DATA_DIR + 'test/')

    batch_size = 64  # Number of sample used simultaneously on a layer
    nb_epochs = 100  # Number of CNN cycles
    kernel_size = 3  # CNN filter size
    pool_size = 2  # Pool size
    kernel_nb_1 = 32  # Number of filters
    kernel_nb_2 = 64  # Number of filters
    dropout_1 = 0.25  # Probability of dropout
    dropout_2 = 0.5  # Probability of dropout
    hidden_size = 64  # Neurons number
    activation = 'relu'  # Activation function
    f_activation = 'sigmoid'  # Final activation function
    nb_classes = 1  # Number of classes -1
    val_split = 0.1  # Percentage of validation data in training data

    # TODO: Zero-center data not needed because we are already rescaling to 0 -> 1 ?
    # x_train = x_train.astype('float64')
    x_test = x_test.astype('float64')
    # x_train -= np.mean(x_train, axis=0)
    # x_train /= 255
    x_test /= 255
    # y_train = to_categorical(y_train, 2)
    # y_test = to_categorical(y_train, 2)

    # Build validation directory for training
    build_train_validation_data(DATA_DIR + 'train/')

    # Build model
    model = Sequential()
    model.add(Conv2D(kernel_nb_1, (kernel_size, kernel_size), input_shape=(128, 128, 3), activation=activation))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))
    model.add(Dropout(dropout_1))

    model.add(Conv2D(kernel_nb_1, (kernel_size, kernel_size), activation=activation))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Conv2D(kernel_nb_2, (kernel_size, kernel_size), activation=activation))
    model.add(MaxPooling2D(pool_size=(pool_size, pool_size)))

    model.add(Flatten())  # Flatten image to a 1D vector
    model.add(Dense(hidden_size, activation=activation))
    model.add(Dropout(dropout_2))
    model.add(Dense(nb_classes, activation=f_activation))

    # Model summary
    model.summary()

    # Optimizer
    optimizer = optimizers.RMSprop(lr=1e-3)

    # Build learning process
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rotation_range=20,
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
        class_mode='binary',
        shuffle=True)

    validation_generator = test_datagen.flow_from_directory(
        DATA_DIR + 'train/validation',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=2000 // batch_size,
        epochs=nb_epochs,
        validation_data=validation_generator,
        validation_steps=800 // batch_size,
        verbose=1)

    # history = model.fit(x_train, y_train,
    #                     batch_size=batch_size,
    #                     epochs=nb_epochs,
    #                     verbose=1,
    #                     validation_split=val_split,
    #                     shuffle=True)

    # Clear validation directory
    undo_train_validation_data(DATA_DIR + 'train/')

    # Save weights
    model.save_weights(WEIGHTS_DIR + 'weights_' + time.strftime("%d-%m-%Y") + '_' + time.strftime('%H:%M') + '.h5')

    loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)

    print('Accuracy: ', accuracy)
    print('Loss: ', loss)

    # Get predictions
    predictions = model.predict(x_test, verbose=0)
    for pred in predictions:
        print(round(pred[0], 2))
    # print(predictions)
    # total_accuracy = get_total_accuracy(y_test, predictions)
    # waldo_accuracy = get_waldo_accuracy(y_test, predictions)
    # print('Total accuracy of: ' + str(total_accuracy) + '%.')
    # print('Waldo accuracy of: ' + str(waldo_accuracy) + '%.')
    plot_loss_accuracy(history)
