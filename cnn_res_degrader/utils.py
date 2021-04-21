import tensorflow as tf


def enable_gpu_if_possible():
    if physical_devices := tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(physical_devices[0], True)
