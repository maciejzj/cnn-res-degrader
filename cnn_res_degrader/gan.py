from tensorflow.keras import layers

import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard)
from cnn_res_degrader.data_loading import (
    Dataset,
    Subset,
    InterpolationMode,
    ProbaDataGenerator,
    ProbaDirectoryScanner,
    ProbaImagePreprocessor,
    ProbaHistEqualizer,
    ProbaHrToLrResizer)
from cnn_res_degrader.metrics import make_ssim_metric
from cnn_res_degrader.models import make_model, Models
from cnn_res_degrader.train import make_training_data
from cnn_res_degrader.test import make_test_data
from matplotlib import pyplot as plt

# Create the discriminator
discriminator = keras.Sequential(
    [
        keras.Input(shape=(126, 126, 1)),
        layers.Conv2D(64, (3, 3), strides=(2, 2),
                      padding="same"),
        layers.LeakyReLU(),
        layers.MaxPool2D(),
        layers.Conv2D(64, (3, 3), strides=(2, 2),
                      padding="same"),
        layers.MaxPool2D(),
        layers.LeakyReLU(),
        layers.Flatten(),
        layers.Dropout(rate=0.5),
        layers.Dense(1, activation='sigmoid'),
    ],
    name="discriminator",
)


# Create the generator
generator = keras.Sequential(
    [
        keras.Input(shape=(378, 378, 1)),
        layers.Conv2D(64, kernel_size=3, padding="same", activation='relu'),
        layers.Conv2D(64, kernel_size=3, strides=3,
                      padding="same", activation='relu'),
        layers.Conv2D(1, kernel_size=3, padding="same", activation='sigmoid'),
    ],
    name="generator",
)


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

        # Sample random points in the latent space
        batch_size = tf.shape(x)[0]
        y_fake = self.generator(x)

        # Combine them with real images
        combined_images = tf.concat([y_fake, y], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.zeros((batch_size, 1)), tf.ones((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            y_pred = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, y_pred)

        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.d_optimizer.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
        )

        # Assemble labels that say "all real images"
        misleading_labels = tf.ones((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            y_pred = self.discriminator(self.generator(x))
            g_loss = self.loss_fn(misleading_labels, y_pred)

        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.g_optimizer.apply_gradients(
            zip(grads, self.generator.trainable_weights))
        return {"d_loss": d_loss, "g_loss": g_loss}

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
        params['load']['limit_per_scene'])

    gan = GAN(discriminator=discriminator, generator=generator)
    gan.compile(
        d_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        g_optimizer=keras.optimizers.Adam(learning_rate=0.0001),
        loss_fn=keras.losses.BinaryCrossentropy(),
    )

    gan.fit(train_ds, epochs=50)
    test_ds = make_test_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'])
    # To limit the execution time, we only train on 100 batches. You can train on
    # the entire dataset. You will need about 20 epochs to get nice results.
    pred = gan(np.expand_dims(test_ds[29][0][16], axis=0))
    plt.figure()
    plt.title('pred inv')
    plt.imshow(1 - pred[0])
    plt.figure()
    plt.title('inv')
    plt.imshow(pred[0])
    plt.figure()
    plt.title('lr')
    plt.imshow(test_ds[29][1][16][1:-1,1:-1,:])
    plt.show()


if __name__ == '__main__':
    main()
