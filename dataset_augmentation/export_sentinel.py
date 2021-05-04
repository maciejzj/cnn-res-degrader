import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
from scipy.ndimage import shift
from skimage import io, exposure
from matplotlib import pyplot as plt

from cnn_res_degrader.models import make_model, Models


SENTINEL_FILE_BIT_RANGE = 'uint16'


def transform_sentinel_dataset_3xlrs(transformation: Callable,
                                     path: Path,
                                     output_suffix:str):
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
            shifted_hr = shift(hr, (*translations[lr_idx], 0))
            lr = transformation(img_as_batch(shifted_hr))[0]
            cropped_lr = crop_border(lr, 1)
            save_sentinel_img(new_lr_dir/f'lr_0{lr_idx}.png', cropped_lr)
            print(progress_iterator)
            progress_iterator += 1


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


def demo_export(transformation: Callable, hr_path: Path):
    hr = load_sentinel_img_as_array(hr_path)
    lr = load_sentinel_img_as_array(hr_path.parent/'lr_3x/lr_00.png')
    lr_deg = transformation(img_as_batch(hr))[0]
    cropped_lr = crop_border(lr_deg, 1).numpy()
    plt.figure()
    plt.title('lr from degradnet')
    plt.imshow(cropped_lr, cmap='gray')
    plt.figure()
    plt.title('lr artificial')
    plt.imshow(lr, cmap='gray')
    plt.figure()
    plt.title('hr')
    plt.imshow(hr, cmap='gray')
    plt.show()
    

def main():
    model_path = Path(sys.argv[1])
    model = make_model(
        Models.SIMPLE_CONV,
        input_shape=(600, 600, 1),
        use_lr_masks=False)
    model.load_weights(model_path)
    
    demo_hr_path = Path(
        'data/'
        'sentinel-2_artificial/'
        'S2B_MSIL1C_20200806T105619_N0209_R094_T30TWP_20200806T121751/'
        '08280x10440/'
        'b8/'
        'hr.png')
    demo_export(model, demo_hr_path)

    sentinel_root_path = Path('data/sentinel-2_artificial/')
    transform_sentinel_dataset_3xlrs(
        model, sentinel_root_path, str(model_path.stem))


if __name__ == '__main__':
    main()
