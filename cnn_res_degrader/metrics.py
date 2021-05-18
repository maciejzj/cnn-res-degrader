import tensorflow as tf


def make_psnr_metric(max_value=1.0):

    def psnr(y_true, y_pred):
        return 1.0 - tf.reduce_mean(
            tf.image.psnr(y_true, y_pred, max_value))

    return psnr


def make_ssim_metric(max_value=1.0):

    def ssim(y_true, y_pred):
        return 1.0 - tf.reduce_mean(
            tf.image.ssim(y_true, y_pred, max_value))

    return ssim
