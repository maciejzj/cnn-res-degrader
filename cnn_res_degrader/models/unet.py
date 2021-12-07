from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .maskable_degrader import MaskableDegrader


class Unet(MaskableDegrader):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 name='Unet',
                 use_lr_masks=False):

        super(Unet, self).__init__(name=name)
        self._use_lr_masks = use_lr_masks
        self._input_shape = input_shape

        self.conv1a = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.conv1b = layers.Conv2D(64, kernel_size=3, padding='same', activation='relu')
        self.pool1 = layers.MaxPool2D(pool_size=3, padding='same')

        self.conv2a = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.conv2b = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.pool2 = layers.MaxPool2D(pool_size=2, padding='same')

        self.conv3a = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.conv3b = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.dropout1 = layers.Dropout(0.5)
        self.pool3 = layers.MaxPool2D(pool_size=2, padding='same')

        self.conv_mida = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        self.conv_midb = layers.Conv2D(512, kernel_size=3, padding='same', activation='relu')
        self.dropout2 = layers.Dropout(0.5)

        self.upsample1 = layers.UpSampling2D(size=(2, 2))
        self.upconv1a = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')
        self.upconv1b = layers.Conv2D(256, kernel_size=3, padding='same', activation='relu')

        self.upsample2 = layers.UpSampling2D(size=(2, 2))
        self.upconv2a = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')
        self.upconv2b = layers.Conv2D(128, kernel_size=3, padding='same', activation='relu')

        self.outconv = layers.Conv2D(1, kernel_size=3, padding='same', activation='sigmoid')

        self(tf.zeros((1, *input_shape)))

    def call(self, x):
        x = self.conv1a(x)
        x = self.conv1b(x)
        x = self.pool1(x)

        x = self.conv2a(x)
        x = self.conv2b(x)
        residual1 = x
        x = self.pool2(x)

        x = self.conv3a(x)
        x = self.conv3b(x)
        x = self.dropout1(x)
        residual2 = x
        x = self.pool3(x)

        x = self.conv_mida(x)
        x = self.conv_midb(x)
        x = self.dropout2(x)

        x = self.upsample1(x)
        x = tf.image.resize(x, (63, 63))
        x = layers.concatenate([x, residual2], axis=3)
        x = self.upconv1a(x)
        x = self.upconv1b(x)

        x = self.upsample2(x)
        x = layers.concatenate([x, residual1], axis=3)
        x = self.upconv2a(x)
        x = self.upconv2b(x)

        x = self.outconv(x)

        return x
