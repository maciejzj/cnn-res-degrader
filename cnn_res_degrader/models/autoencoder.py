from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .maskable_degrader import MaskableDegrader


class Autoencoder(MaskableDegrader):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 name='AutoencoderConv',
                 use_lr_masks=False):

        super(Autoencoder, self).__init__(name=name)
        self._use_lr_masks = use_lr_masks
        self._input_shape = input_shape

        self.conv1 = layers.Conv2D(
            64,
            kernel_size=3,
            padding='same',
            activation='relu')
        self.conv2 = layers.Conv2D(
            64,
            kernel_size=3,
            strides=3,
            padding='same',
            activation='relu')
        self.conv3 = layers.Conv2D(
            64,
            kernel_size=3,
            strides=2,
            padding='same',
            activation='relu')
        self.upsample = layers.UpSampling2D(
            size=(2, 2))
        self.conv4 = layers.Conv2D(
            1,
            kernel_size=3,
            padding='same',
            activation='sigmoid')

        self(tf.zeros((1, *input_shape)))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsample(x)
        x = self.conv4(x)
        return x
