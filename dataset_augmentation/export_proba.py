import argparse
import shutil
import sys
import yaml
from pathlib import Path
from typing import Any, Callable, Dict, Tuple
from PIL import Image

import numpy as np
from skimage import exposure, io, util, transform
from matplotlib import pyplot as plt

from cnn_res_degrader.models import make_model, Models
from cnn_res_degrader.utils import (
    crop_border,
    enable_gpu_if_possible,
    img_as_batch,
    make_comparison_fig)


def transform_proba_dataset_3xlrs(transformation: Callable,
                                  unregistered_proba_path: Path,
                                  registered_proba_path: Path,
                                  output_suffix: str,
                                  add_noise: bool):
    progress_iterator = 0
    
    if add_noise:
        output_suffix += '_n'

    for reg_scene_dir_path in registered_proba_path.glob('*/**/imgset*'):
        scene_name = reg_scene_dir_path.name
        unreg_scene_dir_path = next(unregistered_proba_path.rglob(scene_name))
        new_scene_dir = Path(str(reg_scene_dir_path).replace(
            'proba-v',
            'proba-v_' + output_suffix))
        new_scene_dir.mkdir(parents=True, exist_ok=True)
        ### dodaj cropa!
        shutil.copy(next(unreg_scene_dir_path.glob('hr.png')), new_scene_dir)

        for reg_hr_path in reg_scene_dir_path.glob('HR*.png'):
            reg_hr_img = load_proba_img_as_array(reg_hr_path)
            if add_noise:
                reg_hr_img += np.random.normal(0.0, 0.015, reg_hr_img.shape)
            lr_img = transformation(img_as_batch(reg_hr_img))[0]
            lr_file_name = reg_hr_path.name.replace('HR', 'LR')
            lr_dir = new_scene_dir/'lr'
            lr_dir.mkdir(parents=True, exist_ok=True)
            save_proba_img(lr_dir/lr_file_name, lr_img)

            print('LRs saved:', progress_iterator := progress_iterator + 1)
            sys.stdout.flush()


def load_proba_img_as_array(path: Path) -> np.ndarray:
    img = np.expand_dims(io.imread(path, as_gray=False), axis=2)
    img = np.clip(img, a_min=None, a_max=pow(2, 14))
    img = exposure.rescale_intensity(
        img, in_range='uint14', out_range=(0.0, 1.0))
    return img


def save_proba_img(path: Path, img):
    img_to_save = exposure.rescale_intensity(
        img, in_range=(0.0, 1.0), out_range='uint14')
    io.imsave(path, img_to_save)


def demo_export(transformation: Callable, hr_path: Path, add_noise: bool):
    hr_img = load_proba_img_as_array(hr_path)
    if add_noise:
        hr_img += np.random.normal(0.0, 0.015, hr_img.shape)
      
    lr_img = load_proba_img_as_array(str(hr_path).replace('HR', 'LR'))
    lr_img_deg = np.array(transformation(img_as_batch(hr_img))[0])
    make_comparison_fig(hr_img, lr_img, lr_img_deg, add_resized_lr=True)

    lr_margin = 30
    make_comparison_fig(crop_border(hr_img, lr_margin * 3),
                        crop_border(lr_img, lr_margin),
                        crop_border(lr_img_deg, lr_margin),
                        add_resized_lr=True)
    plt.show()


def make_params(params_path: Path, model_type: Models) -> Dict[str, Any]:
    with open(params_path) as params_file:
        params = yaml.load(params_file, Loader=yaml.FullLoader)[
            model_type.name]
    return params


def main():
    enable_gpu_if_possible()

    parser = argparse.ArgumentParser(description='Export Sentinel-2 dataset.')

    model_selection = parser.add_mutually_exclusive_group()
    model_selection.add_argument('-s', '--simple', action='store_true',
                                 help='Export using simple conv net.')
    model_selection.add_argument('-a', '--autoencoder', action='store_true',
                                 help='Export using autoencoder net.')
    model_selection.add_argument('-g', '--gan', action='store_true',
                                 help='Export using gan net.')
    

    parser.add_argument('-n', '--noise', action='store_true',
                        help='Add noise to HR before augmentation.')
    parser.add_argument('-d', '--demo', action='store_true',
                        help='Don\'t export dataset, demo inference.')
    parser.add_argument('weights_path')

    args = parser.parse_args()

    if args.simple:
        model_type = Models.SIMPLE_CONV
    elif args.autoencoder:
        model_type = Models.AUTOENCODER
    elif args.gan:
        model_type = Models.GAN

    params = make_params(Path('params.yaml'), model_type)
    weights_path = Path(args.weights_path)
    add_noise = args.noise

    if args.weights_path != '':
      model = make_model(
          model_type,
          params['load']['input_shape'],
          use_lr_masks=False)
      model.load_weights(args.weights_path)
    else:
      model = lambda x: np.expand_dims(np.array(Image.fromarray(np.squeeze(x[0])).resize((126, 126), Image.BICUBIC)), axis=0)

    if args.demo:
        demo_hr_path = Path(
            'data/'
            'proba-v_registered_b/'
            'train/'
            'NIR/'
            'imgset0648/'
            'HR000.png')
        demo_export(model, demo_hr_path, add_noise)
    else:
        unregisterd_proba_path = Path('data/proba-v')
        registerd_proba_path = Path('data/proba-v_registered_b')
        if args.weights_path != '':
            suffix = str(weights_path.stem)
        else:
            suffix = 'bicubic'
        transform_proba_dataset_3xlrs(
            model, unregisterd_proba_path, registerd_proba_path, suffix, add_noise)


if __name__ == '__main__':
    main()
  