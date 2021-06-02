import numpy as np
import tensorflow as tf
from matplotlib import figure
from matplotlib import pyplot as plt
from PIL import Image

from cnn_res_degrader.data_loading import hr_shape_to_lr_shape


def enable_gpu_if_possible():
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        tf.config.experimental.set_memory_growth(physical_devices[0], True)


def img_as_batch(img: np.ndarray) -> np.ndarray:
    return np.expand_dims(img, axis=0)


def crop_border(img: np.ndarray, margin: float) -> np.ndarray:
    return img[margin:-margin, margin:-margin]


def make_comparison_fig(hr_img: np.ndarray,
                        lr_img: np.ndarray,
                        pred_img: np.ndarray,
                        add_resized_lr=False) -> figure.Figure:
    pixel_values = np.concatenate(
        (hr_img.ravel(), lr_img.ravel(), pred_img.ravel()))
    max_display_value = pixel_values.mean() + 2 * pixel_values.std()

    if add_resized_lr:
        fig, axs = plt.subplots(1, 4)
        resized = np.array(Image.fromarray(np.squeeze(hr_img, axis=2)).resize(
            hr_shape_to_lr_shape(hr_img.shape)[:-1],
            Image.BICUBIC))

        axs[3].imshow(resized, vmax=max_display_value)
        axs[3].set_title('LR bicubic')
    else:
        fig, axs = plt.subplots(1, 3)

    axs[0].imshow(hr_img, vmax=max_display_value)
    axs[0].set_title('HR')
    axs[1].imshow(lr_img, vmax=max_display_value)
    axs[1].set_title('LR')
    axs[2].imshow(crop_border(pred_img, 1), vmax=max_display_value)
    axs[2].set_title('LR prediction')

    fig.tight_layout()
    return fig
