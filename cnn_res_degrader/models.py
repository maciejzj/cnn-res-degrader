from typing import Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class SimpleConv(keras.Model):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 name='simple_conv',
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

        # 'Dry running' the model builds it, enabling things like `.summary()`
        # and `.load_weights()`
        self(tf.zeros((1, *input_shape)))

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x

    def get_functional(self) -> keras.Model:
        x = keras.Input(shape=self._input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))

    @tf.function
    def train_step(self, data):
        x, y, y_mask = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            if self._use_lr_masks:
                y_pred = tf.boolean_mask(y_pred, y_mask)
                y = tf.boolean_mask(y, y_mask)
            loss = self.compiled_loss(
                y, y_pred, regularization_losses=self.losses)

        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    @tf.function
    def test_step(self, data):
        x, y, y_mask = data
        y_pred = self(x, training=False)
        if self._use_lr_masks:
            y_pred = tf.boolean_mask(y_pred, y_mask)
            y = tf.boolean_mask(y, y_mask)

        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}
