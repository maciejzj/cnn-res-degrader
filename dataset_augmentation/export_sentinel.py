import argparse
import yaml
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np
from scipy.ndimage import shift
from skimage import exposure, io
from matplotlib import pyplot as plt

from cnn_res_degrader.models import make_model, Models
from cnn_res_degrader.utils import (
    crop_border,
    enable_gpu_if_possible,
    img_as_batch,
    make_comparison_fig)


def transform_sentinel_dataset_3xlrs(transformation: Callable,
                                     path: Path,
                                     output_suffix: str):
    progress_iterator = 0

    for hr_path in path.glob('*/**/hr.png'):
        new_hr_dir = Path(str(hr_path.parent).replace(
            'sentinel-2_artificial', 'sentinel-2_degraded_' + output_suffix))
        new_lr_dir = new_hr_dir/'lr_3x'
        new_lr_dir.mkdir(parents=True, exist_ok=True)

        hr = load_sentinel_img_as_array(hr_path)
        cropped_hr = crop_border(hr, 3)

        translations_path = hr_path.parent/'lr_3x/translations.txt'
        translations = load_obj_from_repr_file(translations_path)

        save_obj_to_repr_file(translations, new_hr_dir/'translations.txt')
        save_sentinel_img(new_hr_dir/'hr.png', cropped_hr)

        for lr_idx in translations.keys():
            hr_xy_shift = (-3 * s for s in translations[lr_idx])
            shifted_hr = shift(hr, (*hr_xy_shift, 0))
            lr = transformation(img_as_batch(shifted_hr))[0]
            cropped_lr = crop_border(lr, 1)
            save_sentinel_img(new_lr_dir/f'lr_0{lr_idx}.png', cropped_lr)
            print(progress_iterator := progress_iterator + 1)


def load_sentinel_img_as_array(path: Path) -> np.ndarray:
    img = np.expand_dims(io.imread(path, as_gray=False), axis=2)
    img = np.clip(img, a_min=None, a_max=pow(2, 14))
    img = exposure.rescale_intensity(
        img, in_range='uint14', out_range=(0.0, 1.0))
    return img


def save_sentinel_img(path: Path, img):
    img_to_save = exposure.rescale_intensity(
        img, in_range=(0.0, 1.0), out_range='uint14')
    io.imsave(path, img_to_save)


def load_obj_from_repr_file(path) -> Any:
    with open(path, 'r') as file_:
        ret = eval(file_.readline())
    return ret


def save_obj_to_repr_file(obj: Any, path: Path):
    with open(path, 'w') as file_:
        file_.write(repr(obj))


def demo_export(transformation: Callable, hr_path: Path):
    hr_img = load_sentinel_img_as_array(hr_path)
    lr_img = load_sentinel_img_as_array(hr_path.parent/'lr_3x/lr_00.png')
    lr_img_deg = transformation(img_as_batch(hr_img))[0]
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
                                 help='Train simple conv net.')
    model_selection.add_argument('-a', '--autoencoder', action='store_true',
                                 help='Train autoencoder et.')
    model_selection.add_argument('-g', '--gan', action='store_true',
                                 help='Train gan net.')

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

    model = make_model(
        model_type,
        params['load']['input_shape'],
        use_lr_masks=False)
    model.load_weights(args.weights_path)

    if args.demo:
        demo_hr_path = Path(
            'data/'
            'sentinel-2_artificial/'
            'S2B_MSIL1C_20200806T105619_N0209_R094_T30TWP_20200806T121751/'
            '08280x10440/'
            'b8/'
            'hr.png')
        demo_export(model, demo_hr_path)
    else:
        sentinel_root_path = Path('data/sentinel-2_artificial/')
        suffix = str(weights_path.stem)
        transform_sentinel_dataset_3xlrs(model, sentinel_root_path, suffix)


if __name__ == '__main__':
    main()
