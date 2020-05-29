from datetime import datetime
import numpy as np
import keras
from keras import models
from keras import layers
from keras import losses
from keras import optimizers
from keras import backend as K
from keras.callbacks import TensorBoard
from matplotlib import pyplot as plt
from skimage import io, util

from data_loader import rm_border_from_imgs

keras.backend.set_floatx('float64')


def make_model():

    model = models.Sequential()

    model.add(layers.InputLayer(input_shape=(378, 378, 1)))

    model.add(layers.Conv2D(66, kernel_size=3,
                            padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((2, 2)))

    model.add(layers.Conv2D(32, kernel_size=3,
                            padding='same', activation='relu'))
    model.add(layers.MaxPooling2D((3, 3)))

    model.add(layers.Conv2DTranspose(32, kernel_size=3, padding='same', activation='relu',
                                     strides=2))

    model.add(layers.Conv2D(1, kernel_size=3,
                            padding='same', activation='sigmoid'))

    return model


train, test = np.load("dat-red-one-per-scene.npy", allow_pickle=True)
train_hr, train_lr, _ = train

x = np.array(train_hr).reshape(-1, 378, 378, 1)
y = np.array(train_lr).reshape(-1, 126, 126, 1)

model = make_model()
model.compile(loss=losses.MeanSquaredError(),
              optimizer=optimizers.Adam(),
              metrics=['mae']
              )


model.summary()
model.fit(x, y,
          validation_split=0.2,
          epochs=16,
          callbacks=[TensorBoard(log_dir='log/fit-' +
                                 datetime.now().strftime("%y-%m-%d-%H:%M:%S"))]
          )

p = model.predict(x[0].reshape(1, 378, 378, 1))

plt.figure()
plt.imshow(p.reshape(126, 126), cmap="gray")
plt.colorbar()
plt.figure()
plt.imshow(x[0].reshape(378, 378), cmap="gray")
plt.colorbar()
plt.figure()
plt.imshow(y[0].reshape(126, 126), cmap="gray", norm=None)
plt.colorbar()
plt.show()
