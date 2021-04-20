from typing import Tuple
from enum import Enum, auto

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class Models(Enum):
    SIMPLE_CONV = auto()
    AUTOENCODER_CONV = auto()
    GAN_SIMPLE_CONV = auto()
    GAN_AUTOENCODER_CONV = auto()


def make_model(model: Models, *args, **kwargs):
    model_inits = {Models.SIMPLE_CONV: SimpleConv,
                   Models.AUTOENCODER_CONV: AutoencoderConv,
                   Models.GAN_SIMPLE_CONV: GanSimpleConv,
                   Models.GAN_AUTOENCODER_CONV: GanAutoencoderConv}
    return model_inits[model](*args, **kwargs)


class DegraderModelWithMaskableLoss(keras.Model):
    def __init__(self, name: str):
        super(DegraderModelWithMaskableLoss, self).__init__(name=name)

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

    def get_functional(self) -> keras.Model:
        x = keras.Input(shape=self._input_shape)
        return keras.Model(inputs=[x], outputs=self.call(x))


class SimpleConv(DegraderModelWithMaskableLoss):
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

        self.build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        return x


class AutoencoderConv(DegraderModelWithMaskableLoss):
    def __init__(self,
                 input_shape: Tuple[int, int, int],
                 name='AutoencoderConv',
                 use_lr_masks=False):

        super(AutoencoderConv, self).__init__(name=name)
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

        self.build(input_shape)

    def call(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.upsample(x)
        x = self.conv4(x)
        return x


class GanSimpleConv(keras.Model):
    def __init__(self):
        raise NotImplementedError()


class GanAutoencoderConv(keras.Model):
    def __init__(self):
        raise NotImplementedError()
