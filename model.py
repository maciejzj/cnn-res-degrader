import numpy as np
from keras import models
from keras import layers
from keras import losses
from keras import optimizers

from data_loader import *

hr_shape = (372, 372, 1, 64)

model = models.Sequential()

model.add(layers.InputLayer(input_shape=(372, 372, 1)))

model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())

model.add(layers.Conv2D(64, kernel_size=3, activation='relu'))
model.add(layers.BatchNormalization())
model.add(layers.MaxPooling2D())

model.add(layers.Flatten())
model.add(layers.Dense(49, activation='softmax'))

model.add(layers.Reshape((7, 7, 1)))

model.add(layers.Conv2DTranspose(
    64, kernel_size=3, strides=3, activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2DTranspose(
    64, kernel_size=3, strides=3, activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2DTranspose(
    64, kernel_size=3, strides=2, padding='same', activation='relu'))
model.add(layers.BatchNormalization())

model.add(layers.Conv2D(1, (3, 3), padding='same', activation='relu'))

model.compile(loss=losses.MeanSquaredError(),
              optimizer=optimizers.RMSprop(),
              metrics=['mae'])

model.summary()

train, test = np.load("NIR.npy", allow_pickle=True)
train_hr, train_lr, _ = train
train_hr = rm_border_from_imgs(train_hr)

x = np.array(train_hr).reshape(9910, 372, 372, 1)
y = np.array(train_lr).reshape(9910, 126, 126, 1)

model.fit(x, y, validation_split=0.2, epochs=2)
