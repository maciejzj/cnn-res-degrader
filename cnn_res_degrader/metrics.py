import tensorflow as tf


def make_psnr_loss(max_value=1.0):

    def psnr(y_true, y_pred):
        return 1.0 - tf.reduce_mean(
            tf.image.psnr(y_true, y_pred, max_value))

    return psnr


def make_ssim_loss(max_value=1.0):

    def ssim(y_true, y_pred):
        return 1.0 - tf.reduce_mean(
            tf.image.ssim(y_true, y_pred, max_value))

    return ssim


def make_lpips_loss():

    def lpips(y_true, y_pred):
        raise NotImplementedError()

    return lpips
