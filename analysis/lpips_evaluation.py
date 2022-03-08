import numpy as np
from pathlib import Path
from skimage import data

from cnn_res_degrader.lpips_tf2.models.lpips_tensorflow import learned_perceptual_metric_model

image_size = 512

model_dir = Path('cnn_res_degrader/lpips_tf2/models')
vgg_ckpt_fn = model_dir/'vgg'/'exported'
lin_ckpt_fn = model_dir/'lin'/'exported'

lpips = learned_perceptual_metric_model(image_size, vgg_ckpt_fn, lin_ckpt_fn)

img_rand1 = np.random.uniform(size=(1, image_size, image_size, 1))
img_rand2 = np.random.uniform(size=(1, image_size, image_size, 1))

dist = lpips([img_rand1, img_rand2])
print(f'Distance for random images: {dist}')

dist = lpips([data.brick().reshape(image_size, image_size, 1), data.grass().reshape(image_size, image_size, 1)])
print(f'Distance for img of bricks vs image of grass: {dist}')

dist = lpips([data.grass().reshape(image_size, image_size, 1), data.gravel().reshape(image_size, image_size, 1)])
print(f'Distance for img of grass vs image of gravel: {dist}')

dist = lpips([data.grass().reshape(image_size, image_size, 1), data.grass().reshape(image_size, image_size, 1)])
print(f'Distance for two imgs of grass (same pic): {dist}')
