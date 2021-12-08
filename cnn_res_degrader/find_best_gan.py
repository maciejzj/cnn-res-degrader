import argparse
from pathlib import Path
from typing import Tuple

from natsort import natsorted
from skimage import metrics
from tensorflow import keras

from cnn_res_degrader.utils import enable_gpu_if_possible
from cnn_res_degrader.test import compare, make_test_data
from cnn_res_degrader.train import make_params, make_training_data
from cnn_res_degrader.models import make_model, Models
from cnn_res_degrader.data_loading import (
    Dataset,
    ProbaDataGenerator,
    ProbaDirectoryScanner,
    ProbaImagePreprocessor,
    Subset)


def find_best_weights(model: keras.Model,
                      weights_dir_path: Path,
                      dataset: ProbaDataGenerator) -> Tuple[float, Path]:
    best_ssim = 0
    best_weights_file_path = Path('')

    lr_real = dataset.to_lr_array()

    for weights_file_path in natsorted(weights_dir_path.glob('*.h5')):
        print(f'Processing: {weights_file_path}.')
        model.load_weights(weights_file_path)
        lr_preds = model.predict(dataset, verbose=1)

        ssim_score = compare(lr_preds,
                             lr_real,
                             lambda x, y: metrics.structural_similarity(
                                 x, y, data_range=1., multichannel=True))

        if ssim_score > best_ssim:
            best_ssim = ssim_score
            best_weights_file_path = weights_file_path
            print(f'Found new best SSIM: {best_ssim}.')
        else:
            print(f'SSIM: {ssim_score} not better than best: {best_ssim}.')

        print()

    return best_ssim, best_weights_file_path


def find_best_gan(weights_dir_path: Path):
    params = make_params(Path('params.yaml'), Models.GAN)

    _, val_ds = make_training_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'],
        params['load']['validation_split'],
        {'equalize_hist': False, 'artificial_lr': False},
        params['load']['limit_per_scene'])

    test_ds = make_test_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'])

    val_ds = make_validation_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'],
        params['load']['validation_split'],
        params['load']['limit_per_scene'])

    model = make_model(
        Models.GAN,
        params['load']['input_shape'],
        use_lr_masks=False)

    best_ssim, best_weights_file_path = find_best_weights(
        model, weights_dir_path, val_ds)

    print(
        f'Best SSIM: {best_ssim}, for weights file: {best_weights_file_path}.')


def make_validation_data(
        dataset: Dataset,
        input_shape: Tuple[int, int, int],
        batch_size: int,
        validation_split: float,
        limit_per_scene: int) -> ProbaDataGenerator:

    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v_registered_a'),
        dataset=dataset,
        subset=Subset.TRAIN,
        splits={'train': validation_split, 'val': 1.0 - validation_split},
        limit_per_scene=limit_per_scene)

    val_ds = ProbaDataGenerator(
        dir_scanner.get_split('val'),
        hr_shape=input_shape,
        batch_size=batch_size,
        shuffle=False,
        preprocessor=ProbaImagePreprocessor())

    return val_ds


def main():
    enable_gpu_if_possible()

    parser = argparse.ArgumentParser(
        description='Find best GAN weights in dir.')

    parser.add_argument('weights_dir_path')
    args = parser.parse_args()
    weights_dir_path = Path(args.weights_dir_path)

    find_best_gan(weights_dir_path)


if __name__ == '__main__':
    main()
