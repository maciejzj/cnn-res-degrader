import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.ndimage import shift
from skimage import io, exposure

from cnn_res_degrader.models import make_model, Models


SENTINEL_FILE_BIT_RANGE = 'uint12'


def transform_sentinel_dataset_lrs(transformation: Callable, path: Path):
    for hr_path in path.glob('*/**/hr.png'):
        hr = load_sentinel_img_as_array(hr_path)

        for translations_path in hr_path.parent.glob('lr_3x/translations.txt'):
            translations = load_obj_from_repr_file(translations_path)

            new_lr_dir = Path(str(translations_path.parent).replace(
                'sentinel-2_artificial', 'sentinel-2_degraded'))
            new_lr_dir.mkdir(parents=True, exist_ok=True)

            save_obj_to_repr_file(translations, new_lr_dir/'translations.txt')
            save_sentinel_img(new_lr_dir.parent/'hr.png', hr)

            for lr_idx in translations.keys():
                shifted_hr = shift(hr, (*translations[lr_idx], 0))
                lr = transformation(img_as_batch(shifted_hr))[0]
                cropped_lr = crop_border(lr, 1)
                save_sentinel_img(new_lr_dir/f'lr_0{lr_idx}.png', cropped_lr)


def load_sentinel_img_as_array(path: Path) -> np.ndarray:
    img = np.expand_dims(io.imread(path, as_gray=False), axis=2)
    return exposure.rescale_intensity(
        img, in_range=SENTINEL_FILE_BIT_RANGE, out_range='float64')


def save_sentinel_img(path: Path, img):
    img_to_save = exposure.rescale_intensity(
        img, in_range='float64', out_range=SENTINEL_FILE_BIT_RANGE)
    io.imsave(path, img_to_save)


def load_obj_from_repr_file(path) -> Any:
    with open(path, 'r') as file_:
        ret = eval(file_.readline())
    return ret


def save_obj_to_repr_file(obj: Any, path: Path):
    with open(path, 'w') as file_:
        file_.write(repr(obj))


def img_as_batch(img: np.ndarray) -> np.ndarray:
    return np.expand_dims(img, axis=0)


def crop_border(img: np.ndarray, margin: float) -> np.ndarray:
    return img[margin:-margin, margin:-margin]


def main():
    model = make_model(
        Models.SIMPLE_CONV,
        input_shape=(600, 600, 1),
        use_lr_masks=False)
    model.load_weights(sys.argv[1])

    sentinel_root_path = Path('dataset_augmentation/sentinel-2_artificial')
    transform_sentinel_dataset_lrs(model, sentinel_root_path)


if __name__ == '__main__':
    main()
