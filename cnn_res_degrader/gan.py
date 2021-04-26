import yaml
from pathlib import Path
from typing import Any, Dict

import tensorflow as tf
import numpy as np
from tensorflow import keras
from cnn_res_degrader.data_loading import (
    Dataset,
    InterpolationMode,
    ProbaImagePreprocessor,
    ProbaHistEqualizer,
    ProbaHrToLrResizer)
from cnn_res_degrader.models import Models
from cnn_res_degrader.train import make_training_data
from cnn_res_degrader.test import make_test_data
from cnn_res_degrader.utils import enable_gpu_if_possible

from matplotlib import pyplot as plt


enable_gpu_if_possible()


def make_discriminator() -> keras.Model:
    discriminator = keras.Sequential(
        [
            keras.Input(shape=(126, 126, 1)),
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
            keras.layers.Dropout(rate=0.5),
            keras.layers.Dense(1, activation='sigmoid'),
        ],
        name='discriminator',
    )
    return discriminator


def make_generator() -> keras.Model:
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


class GAN(keras.Model):
    def __init__(self, discriminator, generator):
        super(GAN, self).__init__()
        self.discriminator = discriminator
        self.generator = generator

    def compile(self, d_optimizer, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.d_optimizer = d_optimizer
        self.g_optimizer = g_optimizer
        self.loss_fn = loss_fn

    def train_step(self, data):
        x, y, y_mask = data
        batch_size = tf.shape(x)[0]

        y_fake = self.generator(x)

        # Discriminator training
        discriminator_input = tf.concat([y_fake, y], axis=0)
        fake_labels = tf.zeros((batch_size, 1))
        true_labels = tf.ones((batch_size, 1))
        labels = tf.concat([fake_labels, true_labels], axis=0)
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

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

        return {'d_loss': d_loss, 'g_loss': g_loss}

    def call(self, x):
        x = self.generator(x)
        return x


def make_preprocessor(prep_params: Dict[str, Any]) -> ProbaImagePreprocessor:
    transformations = []

    if prep_params['equalize_hist']:
        transformations.append(ProbaHistEqualizer())

    if prep_params['artificial_lr']:
        transformations.append(ProbaHrToLrResizer(
            prep_params['interpolation_mode']))

    return ProbaImagePreprocessor(*transformations)


def make_params(params_path: Path) -> Dict[str, Any]:
    with open(params_path) as params_file:
        params = yaml.load(params_file, Loader=yaml.FullLoader)

    params['model'] = Models[params['model']]
    params['load']['dataset'] = Dataset[params['load']['dataset']]
    prep = params['load']['preprocess']
    prep['interpolation_mode'] = InterpolationMode[prep['interpolation_mode']]

    return params


def main():
    params = make_params(Path('params.yaml'))

    train_ds, val_ds = make_training_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'],
        params['load']['validation_split'],
        params['load']['preprocess'],
        None)

    gan = GAN(discriminator=make_discriminator(),
              generator=make_generator())
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )
    gan.fit(train_ds, epochs=5)

    test_ds = make_test_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'])

    pred = gan(np.expand_dims(test_ds[29*4][0][7], axis=0))

    plt.figure()
    plt.title('hr')
    plt.imshow(test_ds[29*4][0][7])
    plt.savefig('hr.png', dpi=300)

    plt.figure()
    plt.title('pred')
    plt.imshow(pred[0])
    plt.savefig('pred.png', dpi=300)

    plt.figure()
    plt.title('lr')
    plt.imshow(test_ds[29*4][1][7][1:-1, 1:-1, :])
    plt.savefig('lr.png', dpi=300)


if __name__ == '__main__':
    main()
