import glob
import os
import numpy as np
from skimage import exposure, io, util, img_as_float64, transform
from PIL import Image
import copy


def load_imageset(dataset_dir='./dataset-v11',
                  dataset_name='NIR', limit_per_scene=None):
    if dataset_name not in ('NIR', 'RED'):
        raise RuntimeError('Unknown dataset name')
    else:
        train = load_imageset_from_path(
            os.path.join(
                dataset_dir, 'train', dataset_name),
            limit_per_scene=limit_per_scene)
        test = load_imageset_from_path(
            os.path.join(
                dataset_dir, 'test', dataset_name),
            limit_per_scene=limit_per_scene)
        return (train, test)


def load_imageset_from_path(path, limit_per_scene=None):
    if not os.path.isdir(path):
        raise RuntimeError('Dataset path does not exist')
    else:
        hr, lr, lr_masks = [], [], []
        scenes = get_scenes(path)
        for scene in scenes:
            hr += load_imgs_with_prefix(
                scene, 'HR', limit_per_scene)
            lr += load_imgs_with_prefix(
                scene, 'LR', limit_per_scene)
            lr_masks += load_imgs_with_prefix(
                scene, 'QM', limit_per_scene)
        return (hr, lr, lr_masks)


def get_scenes(path):
    scenes = []
    for root, dirs, _ in os.walk(path):
        for name in sorted(dirs):
            scenes.append(
                os.path.join(root, name))
    return scenes


def load_imgs_with_prefix(path, prefix, limit=None):
    imgs = []
    glob_path = os.path.join(path, prefix)
    img_names = sorted(glob.glob(glob_path + '*'))

    if limit is not None:
        img_names = img_names[:limit]

    for img_name in img_names:
        print(img_name)
        img = io.imread(img_name, as_gray=True)
        img = exposure.rescale_intensity(
            img, in_range='uint14')
        img = img_as_float64(img)
        imgs.append(img)
    return imgs


def store_imgset_as_npy_files(output_file_name, dataset_dir='./dataset-v11',
                              dataset_name='NIR', limit_per_scene=None):
    '''
    Saves as:
        [ [train_hr, train_lr, train_lr_masks],
          [test_hr, test_lr, test_lr_masks] ]
    '''
    dataset = load_imageset(
        dataset_dir=dataset_dir, dataset_name=dataset_name,
        limit_per_scene=limit_per_scene)
    np.save(output_file_name + '.npy', dataset)


def rm_border_from_imgs(imgs, border_width=3):
    ret = []
    crop_borders = ((border_width, border_width),
                    (border_width, border_width))
    for img in imgs:
        img = util.crop(img, crop_borders)
        ret.append(img)
    return ret


def equalize_hist_in_npy_dataset(dataset):
    for i, _ in enumerate(dataset):
        for j in range(0, 2):
            for k, _ in enumerate(dataset[i][j]):
                dataset[i][j][k] = exposure.equalize_hist(
                    dataset[i][j][k])
    return dataset


def transform_y(dataset, tf_func):
    for i, _ in enumerate(dataset):
        for j, _ in enumerate(dataset[i][1]):
            dataset[i][1][j] = tf_func(dataset[i][0][j])
    return dataset


def transform_xy(dataset, tf_func):
    for i, _ in enumerate(dataset):
        for j in range(0, 2):
            for k, _ in enumerate(dataset[i][j]):
                dataset[i][j][k] = tf_func(dataset[i][j][k])
    return dataset


def make_hist_eq_xy_dataset(name_prefix, source_dataset):
    def tf(img): return exposure.equalize_hist(img)
    cp = copy.deepcopy(source_dataset)
    modified = transform_xy(cp, tf)
    np.save(name_prefix + '-eqhist' + '.npy', modified)


def make_resized_y_dataset(name_prefix, source_dataset):
    interpolation_modes = {
        'nearest': Image.NEAREST,
        'bilinear': Image.BILINEAR,
        'bicubic': Image.BICUBIC,
        'lanczos': Image.LANCZOS
    }
    transformations = []
    for name, mode in interpolation_modes.items():
        def tf(img): return np.array(Image.fromarray(img).resize((126, 126), mode))
        cp = copy.deepcopy(source_dataset)
        modified = transform_y(cp, tf)
        np.save(os.path.splitext(name_prefix)[0] + '-' + name + '.npy', modified)


def augment_dataset(dataset_name):
    dataset = np.load(dataset_name, allow_pickle=True)
    make_resized_y_dataset(dataset_name, dataset)
    make_hist_eq_xy_dataset(dataset_name, dataset)


if __name__ == "__main__":
    store_imgset_as_npy_files(
        'data/dat-nir-one-per-scene', dataset_name='NIR', limit_per_scene=1)
    store_imgset_as_npy_files(
        'data/dat-red-one-per-scene', dataset_name='RED', limit_per_scene=1)

    augment_dataset('data/dat-nir-one-per-scene.npy')
    augment_dataset('data/dat-red-one-per-scene.npy')
