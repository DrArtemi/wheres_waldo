from utils import load_data, build_train_validation_data, undo_train_validation_data
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras import optimizers
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


def save_weights(model):
    model.save_weights(WEIGHTS_DIR + 'weights_' + time.strftime("%d-%m-%Y") + '_' + time.strftime('%H:%M') + '.h5')


def build_CNN():
    learning_rate = 1e-3

    # Build model
    model = Sequential()
    model.add(Conv2D(32, (3, 3), input_shape=(128, 128, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(64, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())  # Flatten image to a 1D vector
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    # Model summary
    model.summary()

    # Optimizer
    optimizer = optimizers.RMSprop(lr=learning_rate)

    # Build learning process
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer,
                  metrics=['accuracy'])

    return model


def train_CNN(model, nb_epochs):
    x_train, y_train = load_data(DATA_DIR + 'train')

    x_train = x_train.astype('float64')
    x_train /= 255

    batch_size = 32  # Number of sample used simultaneously on a layer
    val_split = 0.1  # Percentage of validation data in training data

    history = model.fit(x_train, y_train,
                        batch_size=batch_size,
                        epochs=nb_epochs,
                        verbose=1,
                        validation_split=val_split,
                        shuffle=False)

    save_weights(model)

    return model, history


def train_aug_CNN(model, nb_epochs):
    batch_size = 32  # Number of sample used simultaneously on a layer

    # Build validation directory for training
    build_train_validation_data(DATA_DIR + 'train/')

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1. / 255,
        horizontal_flip=True,
        fill_mode='nearest')

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR + 'train/train',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR + 'train/validation',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    history = model.fit_generator(
        train_generator,
        samples_per_epoch=train_generator.samples,
        epochs=nb_epochs,
        validation_data=validation_generator,
        verbose=1)

    # Clear validation directory
    undo_train_validation_data(DATA_DIR + 'train/')

    save_weights(model)

    return model, history


def evaluate_CNN(model, history):
    batch_size = 32  # Number of sample used simultaneously on a layer

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        DATA_DIR + 'test',
        target_size=(128, 128),
        batch_size=batch_size,
        class_mode='binary'
    )

    results = model.evaluate_generator(test_generator)

    print('Loss: ' + str(results[0]))
    print('Accuracy: ' + str(results[1]))

    plot_loss_accuracy(history)


def cnn_algorithm():
    #TODO: Uncomment this if image folder architecture is not normal
    # undo_train_validation_data(DATA_DIR + 'train/')
    model = build_CNN()
    model, history = train_CNN(model, 100)
    # model, history = train_aug_CNN(model, 100)
    evaluate_CNN(model, history)
