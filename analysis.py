from skimage import metrics
from data_loader import *
from model import *
import numpy as np
from skimage import data, img_as_float
from statistics import mean
import matplotlib
import matplotlib.pyplot as plt


def compare(x_set, y_set, compare_func):
    psnrs = []
    for x, y in zip(x_set, y_set):
        psnrs.append(compare_func(img_as_float(x), img_as_float(y)))
    return mean(psnrs)


def predict_with_pretrained_weights(x, model, weight_files_list):
    predictions = []
    for weight_file in weight_files_list:
        model.load_weights(weight_file)
        predictions.append(model.predict(x))
    return predictions


def load_modified_datasets(dataset_path_base, labels):
    datasets = []
    for label in modified_labels:
        _, test = np.load(dataset_path_base + '-' +
                          label + '.npy', allow_pickle=True)
        _, y, _ = test
        datasets.append(np.array(y).reshape(-1, 126, 126, 1))
    return datasets


def make_heatmap_data(imagesets, compare_func):
    heatmap_size = len(imagesets)
    heatmap = np.zeros((heatmap_size, heatmap_size))
    for i, imgs1 in enumerate(imagesets):
        for j, imgs2 in enumerate(imagesets):
            heatmap[i][j] = compare(imgs1, imgs2, compare_func)
    return heatmap


def plot_all_vs_all_heatmap(data, labels, title):
    fig, ax = plt.subplots()

    ax.set_xticks(np.arange(len(labels)))
    ax.set_yticks(np.arange(len(labels)))
    ax.set_xticklabels(labels)
    ax.set_yticklabels(labels)

    plt.setp(ax.get_xticklabels(), rotation=45,
             ha="right", rotation_mode="anchor")

    im = ax.imshow(data)
    ax.figure.colorbar(im, ax=ax)

    for i, lab1 in enumerate(labels):
        for j, lab2 in enumerate(labels):
            ax.text(j, i, round(data[i][j], 3),
                    ha="center", va="center", color="w")

    ax.set_title(title)
    fig.tight_layout()
    plt.show()


if __name__ == '__main__':
    _, _, x, y = make_data('data/dat-nir-one-per-scene.npy')

    sets = [y]
    sets_labels = ['real']

    prediction_labels = ['pred_real', 'pred_eqhist']
    model = make_model()
    preds = predict_with_pretrained_weights(x, model,
        ('log/model-20-07-22-10:54:06-nir-real.h5',
         'log/model-20-07-21-19:19:49-nir-eqhist.h5'))
    sets_labels += prediction_labels
    sets += preds 

    modified_labels = ['bicubic', 'bilinear', 'lanczos', 'nearest']
    modified = load_modified_datasets('data/dat-nir-one-per-scene', modified_labels)
    sets_labels += modified_labels
    sets += modified

    heatmap = make_heatmap_data(sets,
        lambda x, y: metrics.peak_signal_noise_ratio(x, y, data_range=1.))
    plot_all_vs_all_heatmap(heatmap, sets_labels, 'ehh')

    heatmap = make_heatmap_data(sets,
        lambda x, y: metrics.mean_squared_error(x, y))
    plot_all_vs_all_heatmap(heatmap, sets_labels, 'ehh')
