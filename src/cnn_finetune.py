from keras import Sequential, optimizers
from keras.applications.vgg16 import VGG16
from keras.layers import Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from utils import *
import matplotlib.pyplot as plt
import numpy as np
import os

DATA_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../Hey-Waldo/256/'
BOTTLENECK_FEATURES_PATH = os.path.dirname(os.path.realpath(__file__)) + '/../bottleneck_features/'
FT_WEIGHTS_DIR = os.path.dirname(os.path.realpath(__file__)) + '/../finetune_weights/'


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


def evaluate_CNN(model, history):
    _, y_test = load_data(DATA_DIR + 'test/')
    batch_size = 32  # Number of sample used simultaneously on a layer

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    test_generator = test_datagen.flow_from_directory(
        DATA_DIR + 'test',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary'
    )

    predictions = model.predict_generator(test_generator)

    # loss, accuracy = model.evaluate(x_test, y_test, batch_size=batch_size)
    # print('Accuracy: ', accuracy)
    # print('Loss: ', loss)

    # predictions = model.predict(x_test, verbose=0)
    # Print predictions vector
    # for idx, pred in enumerate(predictions):
    #     if y_test[idx] == 1:
    #         print(round(pred[0], 2))
    #         plt.imshow(x_test[idx])
    #         plt.show()
    npred = []
    for pred in predictions:
        print(round(pred[0], 2))
        npred.append(round(pred[0]))
    print(npred)
    total_accuracy = get_total_accuracy(y_test, npred)
    waldo_accuracy = get_waldo_accuracy(y_test, npred)
    print('Total accuracy of: ' + str(total_accuracy) + '%.')
    print('Waldo accuracy of: ' + str(waldo_accuracy) + '%.')
    plot_loss_accuracy(history)


def train_vgg_model():
    model = VGG16(include_top=False, weights='imagenet')
    datagen = ImageDataGenerator(rescale=1. / 255)
    batch_size = 32  # Number of sample used simultaneously on a layer

    train_generator = datagen.flow_from_directory(
        DATA_DIR + 'train/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    validation_generator = datagen.flow_from_directory(
        DATA_DIR + 'train/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode=None,
        shuffle=False)

    bottleneck_features_train = model.predict_generator(train_generator,
                                                        train_generator.samples // batch_size + 1)
    bottleneck_features_validation = model.predict_generator(validation_generator,
                                                             validation_generator.samples // batch_size + 1)

    np.save(open(BOTTLENECK_FEATURES_PATH + 'bottleneck_features_train.npy', 'wb'),
            bottleneck_features_train)
    np.save(open(BOTTLENECK_FEATURES_PATH + 'bottleneck_features_validation.npy', 'wb'),
            bottleneck_features_validation)


def train_fc_model():
    batch_size = 32  # Number of sample used simultaneously on a layer

    train_data = np.load(open(BOTTLENECK_FEATURES_PATH + 'bottleneck_features_train.npy', 'rb'))
    validation_data = np.load(open(BOTTLENECK_FEATURES_PATH + 'bottleneck_features_validation.npy', 'rb'))

    _, y_train = load_data(DATA_DIR + 'train/train/')
    _, y_validation = load_data(DATA_DIR + 'train/validation/')

    model = Sequential()
    model.add(Flatten(input_shape=train_data.shape[1:]))
    model.add(Dense(256, activation='relu'))
    model.add(Dropout(0.5))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=optimizers.RMSprop(lr=1e-4),
                  loss='binary_crossentropy', metrics=['accuracy'])

    model.fit(train_data, y_train,
              epochs=10,
              batch_size=batch_size,
              validation_data=(validation_data, y_validation))
    model.save_weights(FT_WEIGHTS_DIR + 'fc_model_weights.h5')


def train_finetuned_model():
    vgg16_model = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
    batch_size = 32  # Number of sample used simultaneously on a layer

    top_model = Sequential()
    top_model.add(Flatten(input_shape=vgg16_model.output_shape[1:]))
    top_model.add(Dense(256, activation='relu'))
    top_model.add(Dropout(0.5))
    top_model.add(Dense(1, activation='sigmoid'))

    # top_model.load_weights(FT_WEIGHTS_DIR + 'fc_model_weights.h5')

    model = Sequential()
    for layer in vgg16_model.layers:
        model.add(layer)

    model.add(top_model)

    # for layer in model.layers[:10]:
    #     layer.trainable = False

    model.summary()

    model.compile(loss='binary_crossentropy',
                  optimizer=optimizers.RMSprop(lr=1e-5),
                  metrics=['accuracy'])

    train_datagen = ImageDataGenerator(
        shear_range=0.2,
        zoom_range=0.2,
        rescale=1. / 255,
        horizontal_flip=True)

    val_datagen = ImageDataGenerator(rescale=1. / 255)

    train_generator = train_datagen.flow_from_directory(
        DATA_DIR + 'train/train',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    validation_generator = val_datagen.flow_from_directory(
        DATA_DIR + 'train/validation',
        target_size=(224, 224),
        batch_size=batch_size,
        class_mode='binary',
        shuffle=True)

    history = model.fit_generator(
        train_generator,
        steps_per_epoch=train_generator.samples // batch_size + 1,
        epochs=100,
        validation_data=validation_generator,
        validation_steps=validation_generator.samples // batch_size + 1,
        verbose=1)

    return model, history


def cnn_ft_algorithm():
    # TODO: Uncomment this if image folder is fcked up
    # undo_train_validation_data(DATA_DIR + 'train/')

    # Build validation directory for training
    build_train_validation_data(DATA_DIR + 'train/')

    # Train vgg model on data
    # train_vgg_model()

    # Train fc model with vgg bottleneck features on data
    # train_fc_model()

    # Finetune vgg16 and fc model with previously trained weights
    model, history = train_finetuned_model()

    evaluate_CNN(model, history)

    # Destroy validation directory for training
    undo_train_validation_data(DATA_DIR + 'train/')
