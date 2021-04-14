import tensorflow as tf


def make_ssim_metric(max_value=1.0):

    def ssim(y_true, y_pred):
        return 1 - tf.reduce_mean(
            tf.image.ssim(y_true, y_pred, max_value))

    return ssim


def make_psnr_metric(max_value=1.0):

    def psnr(y_true, y_pred):
        return 1 - tf.reduce_mean(
            tf.immake_ssim_metricage.psnr(y_true, y_pred, max_value))

    return psnr
