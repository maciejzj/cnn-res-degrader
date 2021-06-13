import yaml
from pathlib import Path
from typing import Any, Dict, Tuple

import tensorflow as tf
import numpy as np
from tensorflow import keras
from cnn_res_degrader.data_loading import (
    Dataset,
    InterpolationMode,
    ProbaImagePreprocessor,
    ProbaHistEqualizer,
    ProbaHrToLrResizer,
    hr_shape_to_lr_shape)
from cnn_res_degrader.metrics import make_ssim_metric

from matplotlib import pyplot as plt


def make_gan(input_shape: Tuple[int, int, int],
             name='Gan',
             use_lr_masks=False) -> keras.Model:
    if use_lr_masks is True:
        raise NotImplementedError()

    discriminator = make_discriminator(hr_shape_to_lr_shape(input_shape))
    generator = make_generator(input_shape)
    gan = Gan(name=name,
              input_shape=input_shape,
              discriminator=discriminator,
              generator=generator)
    return gan


def make_discriminator(input_shape: Tuple[int, int, int]) -> keras.Model:
    discriminator = keras.Sequential(
        [
            keras.Input(shape=input_shape),
            keras.layers.Conv2D(64,
                                kernel_size=3,
                                strides=2,
                                padding='same'),
            keras.layers.LeakyReLU(),
            keras.layers.MaxPool2D(),
            keras.layers.Conv2D(64,
                                kernel_size=3,
                                strides=2,
                                padding='same'),
            keras.layers.MaxPool2D(),
            keras.layers.LeakyReLU(),
            keras.layers.Flatten(),
            keras.layers.Dense(3),
            keras.layers.LeakyReLU(),
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(1, activation='sigmoid'),
        ],
        name='discriminator',
    )
    return discriminator


def make_generator(input_shape: Tuple[int, int, int]) -> keras.Model:
    generator = keras.Sequential(
        [
            keras.Input(shape=(378, 378, 1)),
            keras.layers.Conv2D(64,
                                kernel_size=3,
                                padding='same',
                                activation='relu'),
            keras.layers.Conv2D(64,
                                kernel_size=3,
                                strides=3,
                                padding='same',
                                activation='relu'),
            keras.layers.Conv2D(1,
                                kernel_size=3,
                                padding='same',
                                activation='sigmoid'),
        ],
        name='generator',
    )
    return generator


class Gan(keras.Model):
    def __init__(self, name, input_shape, discriminator, generator):
        super(Gan, self).__init__(name=name)
        self.discriminator = discriminator
        self.generator = generator
        self.step_counter = 0

        self(tf.zeros((1, *input_shape)))

    def compile(self, loss_fn, d_optimizer, g_optimizer):
        super(Gan, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn
        self.val_loss_fn = make_ssim_metric()

    def train_step(self, data):
        x, y, y_mask = data
        batch_size = tf.shape(x)[0]

        y_fake = self.generator(x)

        # Discriminator training
        if(self.step_counter % 3 == 0):
            discriminator_input = tf.concat([y_fake, y], axis=0)
            fake_labels = tf.zeros((batch_size, 1))
            true_labels = tf.ones((batch_size, 1))
            labels = tf.concat([fake_labels, true_labels], axis=0)
            labels += 0.15 * tf.random.uniform(tf.shape(labels))

            with tf.GradientTape() as tape:
                y_pred = self.discriminator(discriminator_input)
                d_loss = self.loss_fn(labels, y_pred)

            grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
            self.d_optimizer.apply_gradients(
                zip(grads, self.discriminator.trainable_weights))

        # Generator training
        misleading_labels = tf.ones((batch_size, 1))

        with tf.GradientTape() as tape:
            y_pred = self.discriminator(self.generator(x))
            g_loss = self.loss_fn(misleading_labels, y_pred)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))

        self.step_counter += 1

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def test_step(self, data):
        x, y, y_mask = data
        y_pred = self(x, training=False)
        loss = self.val_loss_fn(y, y_pred)
        return {'loss': loss}

    def call(self, x):
        x = self.generator(x, training=False)
        return x
