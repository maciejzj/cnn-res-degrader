from datetime import datetime
import numpy as np
import keras
from keras import models
from keras import layers
from keras import losses
from keras import optimizers
from keras import backend as K
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
from skimage import io, util

from data_loader import rm_border_from_imgs


keras.backend.set_floatx('float64')


def make_data(np_dataset_path):
    train, test = np.load(np_dataset_path, allow_pickle=True)
    train_hr, train_lr, _ = train
    test_hr, test_lr, _ = test

    x_train = np.array(train_hr).reshape(-1, 378, 378, 1)
    y_train = np.array(train_lr).reshape(-1, 126, 126, 1)

    x_test = np.array(test_hr).reshape(-1, 378, 378, 1)
    y_test = np.array(test_lr).reshape(-1, 126, 126, 1)

    return x_train, y_train, x_test, y_test


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

    if tensorboard == True:
        callbacks.append(TensorBoard(log_dir='log/fit-' + train_tim))
    if earlystopping == True:
        callbacks.append(EarlyStopping(monitor='val_loss', mode='min',
                                       min_delta=0.001, patience=5, verbose=1))
    if modelcheckpoint == True:
        callbacks.append(ModelCheckpoint('log/model-' + train_tim + '.h5',
                                         monitor='val_loss', mode='min',
                                         save_best_only=True, verbose=1))
    return callbacks


if __name__ == '__main__':
    x_train, y_train, x_test, y_test = make_data(
        'data/dat-nir-one-per-scene.npy')
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

    print(model.evaluate(x_test, y_test))

    # Demo
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
