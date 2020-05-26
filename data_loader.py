import glob
import os
import numpy as np
from skimage import io, util


def load_imageset(dataset_dir='./dataset-v11', dataset_name='NIR'):
    if dataset_name not in ('NIR', 'RED'):
        raise RuntimeError('Unknown dataset name, will abort')
    else:
        train = load_imageset_from_path(
            os.path.join(dataset_dir, 'train', dataset_name))
        test = load_imageset_from_path(
            os.path.join(dataset_dir, 'test', dataset_name))
        return (train, test)


def load_imageset_from_path(path):
    if not os.path.isdir(path):
        raise RuntimeError('Dataset path does not exist')
    else:
        hr, lr, lr_masks = [], [], []
        scenes = get_scenes(path)
        for scene in scenes:
            lr += load_imgs_with_prefix(scene, 'LR')
            lr_masks += load_imgs_with_prefix(scene, 'QM')
            hr += load_imgs_with_prefix(scene, 'HR')
        return (hr, lr, lr_masks)


def get_scenes(path):
    scenes = []
    for root, dirs, _ in os.walk(path):
        for name in sorted(dirs):
            scenes.append(os.path.join(root, name))
    return scenes


def load_imgs_with_prefix(path, prefix):
    imgs = []
    glob_path = os.path.join(path, prefix)
    img_names = glob.glob(glob_path + '*')

    for img_name in sorted(img_names):
        print(img_name)
        imgs.append(io.imread(img_name, as_gray=True))
    return imgs


def rm_border_from_imgs(imgs, border_width=3):
    for img in imgs:
        img = util.crop(img, ((3, 3), (3, 3)))
    return imgs


def store_imgset_as_npy_files(dataset_dir='./dataset-v11', dataset_name='NIR'):
    '''
    Saves as:
        [ [train_hr, train_lr, train_lr_masks],
          [test_hr, test_lr, test_lr_masks] ]
    '''
    dataset = load_imageset(dataset_dir=dataset_dir, dataset_name=dataset_name)
    np.save(dataset_name + ".npy", dataset)


if __name__ == "__main__":
    store_imgset_as_npy_files(dataset_name="NIR")
    store_imgset_as_npy_files(dataset_name="RED")
