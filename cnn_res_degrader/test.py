import argparse
import yaml
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
from matplotlib import pyplot as plt
from skimage import img_as_float, metrics

from cnn_res_degrader.data_loading import (
    hr_shape_to_lr_shape,
    Dataset,
    InterpolationMode,
    ProbaDataGenerator,
    ProbaDirectoryScanner,
    ProbaHrToLrResizer,
    ProbaImagePreprocessor,
    SampleEl,
    SampleTransformation,
    Subset)
from cnn_res_degrader.models import make_model, Models
from cnn_res_degrader.utils import (
    crop_border,
    enable_gpu_if_possible,
    make_comparison_fig)


CompareFunc = Callable[[np.ndarray, np.ndarray], float]


def make_test_data(dataset: Dataset,
                   input_shape: Tuple[int, int, int],
                   batch_size: int) -> ProbaDataGenerator:
    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v_registered_a'),
        dataset=dataset,
        shuffle=False,
        subset=Subset.TEST)

    print(dir_scanner)

    test_ds = ProbaDataGenerator(
        dir_scanner.paths,
        hr_shape=input_shape,
        batch_size=batch_size,
        shuffle=False,
        preprocessor=ProbaImagePreprocessor())

    return test_ds


def make_artificial_datasets(
        load_params: Dict[str, Any]) -> List[SampleTransformation]:
    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v_registered_a'),
        dataset=load_params['dataset'],
        shuffle=False,
        subset=Subset.TEST)

    lr_shape = hr_shape_to_lr_shape(load_params['input_shape'])
    preps = [ProbaImagePreprocessor(ProbaHrToLrResizer(
        mode, target_shape=lr_shape)) for mode in InterpolationMode]
    make_artificial_data_gen = partial(
        ProbaDataGenerator,
        dir_scanner.paths,
        hr_shape=load_params['input_shape'],
        shuffle=False)
    return [make_artificial_data_gen(preprocessor=prep) for prep in preps]


def compare(x_set: np.ndarray,
            y_set: np.ndarray,
            compare_func: CompareFunc) -> float:
    psnrs: List[float] = []
    for x, y in zip(x_set, y_set):
        psnrs.append(compare_func(img_as_float(x), img_as_float(y)))
    return mean(psnrs)


def make_heatmap_data(imagesets: List[np.ndarray],
                      compare_func: CompareFunc) -> np.ndarray:

    heatmap_size = len(imagesets)
    heatmap = np.zeros((heatmap_size, heatmap_size))
    for i, imgs1 in enumerate(imagesets):
        for j, imgs2 in enumerate(imagesets):
            if i == j:
                continue
            heatmap[i][j] = compare(imgs1, imgs2, compare_func)
    return heatmap


def plot_all_vs_all_heatmap(data: np.ndarray,
                            labels: List[str],
                            title: str) -> plt.Figure:
    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    im = ax.imshow(data)
    ax.figure.colorbar(im, ax=ax)

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, round(data[i][j], 3),
                    ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    return fig


def make_heatmaps(lr_sets: List[np.ndarray], labels: List[str]) -> Dict[str, plt.Figure]:
    if len(lr_sets) != len(labels):
        raise ValueError('Got different number of sets and labels')

    figs = {}

    heatmap = make_heatmap_data(
        lr_sets,
        lambda x, y: metrics.peak_signal_noise_ratio(x, y, data_range=1.))
    fig = plot_all_vs_all_heatmap(heatmap, labels, 'PSNR (larger is better)')
    figs['psnr'] = fig

    heatmap = make_heatmap_data(
        lr_sets,
        lambda x, y: metrics.structural_similarity(
            x, y, data_range=1., multichannel=True))
    fig = plot_all_vs_all_heatmap(heatmap, labels, 'SSIM (larger is better)')
    figs['ssim'] = fig

    heatmap = make_heatmap_data(
        lr_sets,
        metrics.mean_squared_error)
    fig = plot_all_vs_all_heatmap(heatmap * 10e3, labels,
                                  'MSE 10e3 (smaller is better)')
    figs['mse'] = fig
    return figs


def extract_name_from_weights_path(path: Path) -> str:
    name_offset = 6
    return path.stem[name_offset:]


def make_params(params_path: Path, model_type: Models) -> Dict[str, Any]:
    with open(params_path) as params_file:
        params = yaml.load(params_file, Loader=yaml.FullLoader)[
            model_type.name]

    params['load']['dataset'] = Dataset[params['load']['dataset']]

    return params


def test(model_type: Models, weights_path: Path, output_dir: Path):
    params = make_params(Path('params.yaml'), model_type)
    name = extract_name_from_weights_path(weights_path)

    model = make_model(
        model_type,
        params['load']['input_shape'],
        use_lr_masks=False)
    model.load_weights(weights_path)

    output_dir = output_dir/f'test-{name}'
    output_dir.mkdir(parents=True, exist_ok=True)

    test_ds = make_test_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'])

    lr_sets_labels = ['real']
    lr_sets = [test_ds.to_lr_array()]

    lr_prediction_labels = ['pred']
    print('Running test inference:')
    lr_preds = model.predict(test_ds, verbose=1)
    lr_sets_labels += lr_prediction_labels
    lr_sets.append(lr_preds)

    print('Creating artificial datasets.')
    artificial_dss = make_artificial_datasets(params['load'])
    lr_artificial_labels = [mode.name for mode in InterpolationMode]
    lr_artificials = [ads.to_lr_array() for ads in artificial_dss]
    lr_sets_labels += lr_artificial_labels
    lr_sets += lr_artificials

    print('Creating metrics heatmaps.')
    figs = make_heatmaps(lr_sets, lr_sets_labels)
    for fig_key in figs:
        figs[fig_key].savefig(
            output_dir/f'{fig_key}_heatmap-{name}.png', dpi=300)

    batch, sample_in_batch = 0, 0
    hr_img = test_ds[batch][SampleEl.HR][sample_in_batch]
    lr_img = test_ds[batch][SampleEl.LR][sample_in_batch]
    pred_img = lr_preds[batch * params['load']['batch_size'] + sample_in_batch]

    print('Creating comparison figures.')
    fig = make_comparison_fig(hr_img, lr_img, pred_img, add_resized_lr=True)
    fig.savefig(output_dir/f'test_preview-{name}.png', dpi=300)

    lr_margin = 30
    fig = make_comparison_fig(crop_border(hr_img, lr_margin * 3),
                              crop_border(lr_img, lr_margin),
                              crop_border(pred_img, lr_margin),
                              add_resized_lr=True)
    fig.savefig(output_dir/f'test_preview_zoomed-{name}.png', dpi=300)
    print('Figures generation done, saving.')


def main():
    enable_gpu_if_possible()

    parser = argparse.ArgumentParser(description='Train networks.')

    model_selection = parser.add_mutually_exclusive_group()
    model_selection.add_argument('-s', '--simple', action='store_true',
                                 help='Train simple conv net.')
    model_selection.add_argument('-a', '--autoencoder', action='store_true',
                                 help='Train autoencoder et.')
    model_selection.add_argument('-g', '--gan', action='store_true',
                                 help='Train gan net.')

    parser.add_argument('weights_path')
    parser.add_argument('output_dir')

    args = parser.parse_args()
    weights_path = Path(args.weights_path)
    output_dir = Path(args.output_dir)

    if args.simple:
        test(Models.SIMPLE_CONV, weights_path, output_dir)
    elif args.autoencoder:
        test(Models.AUTOENCODER, weights_path, output_dir)
    elif args.gan:
        test(Models.GAN, weights_path, output_dir)


if __name__ == '__main__':
    main()
