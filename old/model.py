#!/usr/bin/env python3

'''
Model and training for image shrinking neural network, when executed as script
trains using data loaded from path to npy dataset passed as arg
'''

import argparse
from datetime import datetime

import numpy as np
from keras import layers, losses, models, optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt

K.set_floatx('float64')


def make_data(np_dataset_path):
    (train_dat, test_dat) = np.load(np_dataset_path, allow_pickle=True)
    (train_hr, train_lr, _) = train_dat
    (test_hr, test_lr, _) = test_dat

    x_train = np.array(train_hr).reshape(-1, 378, 378, 1)
    y_train = np.array(train_lr).reshape(-1, 126, 126, 1)

    x_test = np.array(test_hr).reshape(-1, 378, 378, 1)
    y_test = np.array(test_lr).reshape(-1, 126, 126, 1)

    return (x_train, y_train, x_test, y_test)


def make_model():
    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=(378, 378, 1)))
    model.add(layers.Conv2D(64, kernel_size=3,
                            padding='same', activation='relu'))
    model.add(layers.Conv2D(64, kernel_size=5, strides=3,
                            padding='same', activation='relu'))
    model.add(layers.Conv2D(1, kernel_size=3,
                            padding='same', activation='sigmoid'))
    return model


def make_callbacks(tensorboard=True, earlystopping=True, modelcheckpoint=True):
    callbacks = []

    train_tim = datetime.now().strftime("%y-%m-%d-%H:%M:%S")

    if tensorboard:
        callbacks.append(TensorBoard(log_dir='log/fit-' + train_tim))

    if earlystopping:
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min',
                                       min_delta=0.001, patience=5, verbose=1))
    if modelcheckpoint:
        callbacks.append(ModelCheckpoint('log/model-' + train_tim + '.h5',
                                         monitor='val_loss', mode='min',
                                         save_best_only=True, verbose=1))
    return callbacks


def train(x_train, y_train):
    model = make_model()
    callbacks = make_callbacks(earlystopping=False)

    model.compile(loss=losses.MeanSquaredError(),
                  optimizer=optimizers.Adam(),
                  metrics=['mae'])

    model.summary()

    model.fit(x_train, y_train,
              validation_split=0.2,
              epochs=1,
              callbacks=callbacks)
    return model


def main(dataset_path):
    x_train, y_train, x_test, y_test = make_data(dataset_path)
    model = train(x_train, y_train)

    print(model.evaluate(x_test, y_test))

    p = model.predict(x_test[[1]])
    plt.imshow(p.reshape(126, 126), cmap="gray")
    plt.title("Prediction")

    plt.figure()
    plt.imshow(x_test[[1]].reshape(378, 378), cmap="gray")
    plt.title("Input")

    plt.figure()
    plt.imshow(y_test[[1]].reshape(126, 126), cmap="gray", norm=None)
    plt.title("Target")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Train default image shrinking neural network model')

    parser.add_argument(
        'dataset_path',
        help='path to npy dataset which will be used for training')

    args = parser.parse_args()
    main(args.dataset_path)
