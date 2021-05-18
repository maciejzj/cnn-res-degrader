from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .maskable_degrader import MaskableDegrader

class SimpleConv(MaskableDegrader):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 name='SimpleConv',
                 use_lr_masks=False):

        super(SimpleConv, self).__init__(name=name)
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
            1,
            kernel_size=3,
            padding='same',
            activation='sigmoid')

        self(tf.zeros((1, *input_shape)))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

