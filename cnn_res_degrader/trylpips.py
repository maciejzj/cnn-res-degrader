from cnn_res_degrader.lpips_tf2.models.lpips_tensorflow import learned_perceptual_metric_model

import os
import numpy as np
import tensorflow as tf
from PIL import Image

image_size = 64
model_dir = './cnn_res_degrader/lpips_tf2/models'
vgg_ckpt_fn = os.path.join(model_dir, 'vgg', 'exported')
lin_ckpt_fn = os.path.join(model_dir, 'lin', 'exported')
lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

image1 = np.random.uniform(size=(1, image_size, image_size, 3))
image2 = np.random.uniform(size=(1, image_size, image_size, 3))

dist01 = lpips([image1, image2])
print('Distance: {:.3f}'.format(dist01))