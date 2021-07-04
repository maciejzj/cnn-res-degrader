from typing import Tuple
from enum import Enum, auto

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .simple_conv import SimpleConv
from .autoencoder import Autoencoder
from .gan import make_gan


class Models(Enum):
    SIMPLE_CONV = auto()
    AUTOENCODER= auto()
    GAN = auto()


def make_model(model: Models, *args, **kwargs):
    model_inits = {Models.SIMPLE_CONV: SimpleConv,
                   Models.AUTOENCODER: Autoencoder,
                   Models.GAN: make_gan}
    return model_inits[model](*args, **kwargs)
