from pathlib import Path

import tensorflow as tf

from cnn_res_degrader.lpips_tf2.models.lpips_tensorflow import learned_perceptual_metric_model


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
    img_size = 126
    checkpoints_dir_path = Path('cnn_res_degrader/lpips_tf2/models')
    vgg_checkpoints_path = checkpoints_dir_path/'vgg'/'exported'
    lin_checkpoints_path = checkpoints_dir_path/'lin'/'exported'
    lpips_fn = learned_perceptual_metric_model(
        img_size, 
        vgg_checkpoints_path,
        lin_checkpoints_path)

    def lpips(y_true, y_pred):
        y_true_rgb = tf.image.grayscale_to_rgb(y_true)
        y_pred_rgb = tf.image.grayscale_to_rgb(y_pred)
        return lpips_fn([y_true_rgb, y_pred_rgb])

    return lpips
