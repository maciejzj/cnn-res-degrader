import glob
import keras
import os
from skimage import io


def load_imageset(dataset_dir='./dataset-v11', dataset='NIR'):
    if dataset not in ('NIR', 'RED'):
        raise RuntimeError('Unknown dataset type name, will abort')
    else:
        train = load_imageset_from_path(
            os.path.join(dataset_dir, 'train', dataset))
        test = load_imageset_from_path(
            os.path.join(dataset_dir, 'test', dataset))
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


train, test = load_imageset()
train_lr, train_lr_masks, train_hr = train
test_lr, test_lr_masks, test_hr = test
