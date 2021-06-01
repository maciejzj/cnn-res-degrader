import argparse
from pathlib import Path
from typing import Tuple

from natsort import natsorted
from skimage import metrics
from tensorflow import keras

from cnn_res_degrader.utils import enable_gpu_if_possible
from cnn_res_degrader.test import make_params, make_test_data, compare
from cnn_res_degrader.models import make_model, Models
from cnn_res_degrader.data_loading import ProbaDataGenerator


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
            print(f'Found new best SSIM: {best_ssim}')
        
        print()

    return best_ssim, best_weights_file_path


def find_best_gan(weights_dir_path: Path):
    params = make_params(Path('params.yaml'), Models.GAN)

    test_ds = make_test_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'])

    model = make_model(
        Models.GAN,
        params['load']['input_shape'],
        use_lr_masks=False)

    best_ssim, best_weights_file_path = find_best_weights(
        model, weights_dir_path, test_ds)

    print(
        f'Best SSIM: {best_ssim}, for weights file: {best_weights_file_path}.')


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
