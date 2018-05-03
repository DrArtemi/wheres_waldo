from keras import Sequential, optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from utils import *
import os

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/128/'
BOTTLENECK_FEATURES_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../bottleneck_features/'
FT_WEIGHTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../finetune_weights/'


def plot_loss_accuracy(history):
    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 2, 1)
    ax.plot(history.history["loss"], 'r', label="Train Loss")
    ax.plot(history.history["val_loss"], 'b', label="Validation Loss")
    ax.legend()
    ax.set_title('cross_entropy loss')
    ax.grid(True)

    ax = fig.add_subplot(1, 2, 2)
    ax.plot(history.history["acc"], 'r', label="Train Accuracy")
    ax.plot(history.history["val_acc"], 'b', label="Validation Accuracy")
    ax.legend()
    ax.set_title('accuracy')
    ax.grid(True)

    plt.savefig('loss_acc_ft.png')


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


def train_finetuned_model():
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape=(128, 128, 3))
    batch_size = 32  # Number of sample used simultaneously on a layer
    epochs = 100

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    # top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    model = Sequential()
    for layer in vgg16_model.layers:
        layer.trainable = False
        model.add(layer)
    model.add(top_model)

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=2e-5),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        rotation_range=40,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        rescale=1. / 255,
        fill_mode="nearest")

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
        steps_per_epoch=train_generator.samples // batch_size + 1,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size + 1,
        verbose=1)

    save_weights(model)

    return model, history


def cnn_ft_algorithm():
    # TODO: Uncomment this if image folder architecture is not normal
    # undo_train_validation_data(DATA_DIR + 'train/')

    # Build validation directory for training
    build_train_validation_data(DATA_DIR + 'train/')

    # Finetune vgg16 and fc model with previously trained weights
    model, history = train_finetuned_model()

    # Destroy validation directory for training
    undo_train_validation_data(DATA_DIR + 'train/')

    evaluate_CNN(model, history)
