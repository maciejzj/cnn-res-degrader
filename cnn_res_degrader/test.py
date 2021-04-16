import sys
import yaml
from functools import partial
from pathlib import Path
from statistics import mean
from typing import Any, Callable, Dict, List

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
from cnn_res_degrader.models import SimpleConv


CompareFunc = Callable[[np.ndarray, np.ndarray], float]


def make_test_data(load_params: Dict[str, Any]) -> ProbaDataGenerator:
    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v11_shifted'),
        dataset=load_params['dataset'],
        shuffle=False,
        subset=Subset.TEST)

    test_ds = ProbaDataGenerator(
        dir_scanner.paths,
        hr_shape=load_params['input_shape'],
        shuffle=False,
        preprocessor=ProbaImagePreprocessor())

    return test_ds


def make_artificial_datasets(
        load_params: Dict[str, Any]) -> List[SampleTransformation]:
    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v11_shifted'),
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


def demo_heatmaps(lr_sets: List[np.ndarray], labels: List[str]):
    if len(lr_sets) != len(labels):
        raise ValueError('Got different number of sets and labels')

    heatmap = make_heatmap_data(
        lr_sets,
        lambda x, y: metrics.peak_signal_noise_ratio(x, y, data_range=1.))
    plot_all_vs_all_heatmap(heatmap, labels, 'PSNR (larger is better)')

    heatmap = make_heatmap_data(
        lr_sets,
        lambda x, y: metrics.structural_similarity(
            x, y, data_range=1., multichannel=True))
    plot_all_vs_all_heatmap(heatmap, labels, 'SSIM (larger is better)')

    heatmap = make_heatmap_data(
        lr_sets,
        metrics.mean_squared_error)
    plot_all_vs_all_heatmap(heatmap * 10e3, labels,
                            'MSE 10e3 (smaller is better)')


def make_params(params_path: Path) -> Dict[str, Any]:
    with open(params_path) as params_file:
        params = yaml.load(params_file, Loader=yaml.FullLoader)
    params['load']['dataset'] = Dataset[params['load']['dataset']]
    return params


def main():
    params = make_params(Path('params.yaml'))

    test_ds = make_test_data(params['load'])

    lr_sets_labels = ['real']
    lr_sets = [test_ds.to_lr_array()]

    model = SimpleConv(params['load']['input_shape'])
    model.load_weights(sys.argv[1])
    lr_prediction_labels = ['pred_eqhist']
    lr_preds = model.predict(test_ds)
    lr_sets_labels += lr_prediction_labels
    lr_sets.append(lr_preds)

    artificial_dss = make_artificial_datasets(params['load'])
    lr_artificial_labels = [mode.name for mode in InterpolationMode]
    lr_artificials = [ads.to_lr_array() for ads in artificial_dss]
    lr_sets_labels += lr_artificial_labels
    lr_sets += lr_artificials

    demo_heatmaps(lr_sets, lr_sets_labels)

    batch, sample_in_batch = 0, 0
    plt.figure()
    plt.title('HR')
    plt.imshow(test_ds[batch][SampleEl.HR][sample_in_batch])
    plt.figure()
    plt.title('LR')
    plt.imshow(test_ds[batch][SampleEl.LR][sample_in_batch])
    plt.figure()
    plt.title('LR_PRED')
    plt.imshow(lr_preds[0])

    plt.show()


if __name__ == '__main__':
    main()
