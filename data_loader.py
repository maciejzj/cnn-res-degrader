import glob
import os
import numpy as np
from skimage import exposure, io, util, img_as_float64, transform


def load_imageset(dataset_dir='./dataset-v11', dataset_name='NIR', limit_per_scene=None):
    if dataset_name not in ('NIR', 'RED'):
        raise RuntimeError('Unknown dataset name, will abort')
    else:
        train = load_imageset_from_path(
            os.path.join(dataset_dir, 'train', dataset_name),
            limit_per_scene=limit_per_scene)
        test = load_imageset_from_path(
            os.path.join(dataset_dir, 'test', dataset_name),
            limit_per_scene=limit_per_scene)
        return (train, test)


def load_imageset_from_path(path, limit_per_scene=None):
    if not os.path.isdir(path):
        raise RuntimeError('Dataset path does not exist')
    else:
        hr, lr, lr_masks = [], [], []
        scenes = get_scenes(path)
        for scene in scenes:
            hr += load_imgs_with_prefix(scene, 'HR', limit_per_scene)
            lr += load_imgs_with_prefix(scene, 'LR', limit_per_scene)
            lr_masks += load_imgs_with_prefix(scene, 'QM', limit_per_scene)
        return (hr, lr, lr_masks)


def get_scenes(path):
    scenes = []
    for root, dirs, _ in os.walk(path):
        for name in sorted(dirs):
            scenes.append(os.path.join(root, name))
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
        img = exposure.rescale_intensity(img, in_range='uint14')
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
    np.save(output_file_name + ".npy", dataset)


def rm_border_from_imgs(imgs, border_width=3):
    ret = []
    crop_borders = ((border_width, border_width), (border_width, border_width))
    for img in imgs:
        img = util.crop(img, crop_borders)
        ret.append(img)
    return ret


def equalize_hist_in_npy_dataset(dataset):
    for i, _ in enumerate(dataset):
        for j in range(0, 2):
            for k, _ in enumerate(dataset[i][j]):
                dataset[i][j][k] = exposure.equalize_hist(dataset[i][j][k])
    return dataset


def make_y_downscaled(dataset):
    for i, _ in enumerate(dataset):
        for j, _ in enumerate(dataset[i][1]):
            dataset[i][1][j] = transform.resize(dataset[i][0][j], (126, 126))
    return dataset


if __name__ == "__main__":
    store_imgset_as_npy_files("dat-nir", dataset_name="NIR")
    store_imgset_as_npy_files("dat-red", dataset_name="RED")
    store_imgset_as_npy_files("dar-nir-one-per-scene", dataset_name="NIR",
                              limit_per_scene=1)
    store_imgset_as_npy_files("dat-red-one-per-scene", dataset_name="RED",
                              limit_per_scene=1)
