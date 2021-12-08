import math
import random
from enum import Enum, IntEnum
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from skimage import exposure, io, img_as_float
from tensorflow import keras


Sample = Tuple[np.ndarray, np.ndarray, np.ndarray]
SamplePaths = Tuple[Path, Path, Path]
SampleTransformation = Callable[[Sample], Sample]


class SampleEl(IntEnum):
    HR = 0
    LR = 1
    LR_MASK = 2


class Dataset(Enum):
    NIR = 'NIR'
    RED = 'RED'


class Subset(Enum):
    TRAIN = 'train'
    TEST = 'test'


class ProbaDirectoryScanner:
    def __init__(self,
                 path: Path,
                 dataset: Dataset,
                 subset: Subset,
                 shuffle=True,
                 seed: int = None,
                 splits: Dict[str, float] = None,
                 limit_per_scene: int = None):
        self._path = path
        self._subset = subset
        self._dataset = dataset
        self._splits = splits
        self._limit_per_scene = limit_per_scene

        self._paths: List[SamplePaths]
        self._paths = self._collect_data_paths()

        if shuffle is True:
            self._shuffle_paths(seed)

        self._split_slices: Optional[Dict[str, List[SamplePaths]]]
        self._split_slices = None if splits is None else self._split_paths(
            splits)

    def __len__(self):
        return len(self._paths)

    def __str__(self):
        msg = (f'Proba scanner for directory: {self._path.absolute()}\n'
               f'Found: {len(self._paths)} samples.\n')
        if self._split_slices is not None:
            msg += 'Has slices:'
            for split_name in self._split_slices:
                msg += (f'\n\t{split_name}: {len(self._split_slices[split_name])} '
                        'samples.')
        return msg

    @property
    def paths(self) -> List[SamplePaths]:
        return self._paths

    def get_split(self, split_name: str) -> List[SamplePaths]:
        if self._split_slices is None:
            raise RuntimeError('Splice requested, but dataset was not split '
                               'during initialization.')

        return self._split_slices[split_name]

    def _collect_data_paths(self):
        ret = []
        scenes_path = self._path/self._subset.value/self._dataset.value
        for scenedir in sorted(scenes_path.iterdir()):
            if scenedir.is_dir():
                ret += self._collect_data_paths_in_scene(scenedir)

        return ret

    def _collect_data_paths_in_scene(self, scenedir: Path):
        ret = []
        hrs = sorted(scenedir.glob('HR*.png'))[:self._limit_per_scene]
        lrs = sorted(scenedir.glob('LR*.png'))[:self._limit_per_scene]
        lr_masks = sorted(scenedir.glob('QR*.png'))[:self._limit_per_scene]

        for sample_path in zip(hrs, lrs, lr_masks):
            ret.append(sample_path)
        return ret

    def _shuffle_paths(self, seed: int = None):
        random.Random(seed).shuffle(self._paths)

    def _split_paths(
            self, splits: Dict[str, float]) -> Dict[str, List[SamplePaths]]:
        ret = {}
        if not math.isclose(sum(splits.values()), 1.0):
            raise ValueError('Ratios of given splits don\'t add up to 1.')

        split_beg = 0
        for key in splits:
            split_end = split_beg + math.ceil(splits[key] * len(self._paths))
            ret[key] = self._paths[split_beg:split_end]
            split_beg = split_end
        return ret


class ProbaHistMatcher:
    def __call__(self, sample: Sample) -> Sample:
        new_hr = sample[SampleEl.HR]
        new_lr = sample[SampleEl.LR]
        new_hr = exposure.match_histograms(new_hr, new_lr)
        return new_hr, new_lr, sample[SampleEl.LR_MASK]


class InterpolationMode(IntEnum):
    NEAREST = Image.NEAREST
    BILINEAR = Image.BILINEAR
    BICUBIC = Image.BICUBIC
    LANCZOS = Image.LANCZOS


class ProbaHrToLrResizer:
    def __init__(self, interpolation_mode: InterpolationMode):
        self._interpolation_mode = interpolation_mode

    def __call__(self, sample: Sample) -> Sample:
        hr = sample[SampleEl.HR]
        target_shape = hr_shape_to_lr_shape(hr.shape)
        resized = np.array(Image.fromarray(np.squeeze(hr, axis=2)).resize(
            target_shape[:-1],
            self._interpolation_mode))
        resized = np.expand_dims(resized, axis=2)
        return sample[SampleEl.HR], resized, sample[SampleEl.LR_MASK]


class ProbaImagePreprocessor:
    def __init__(self, *args: SampleTransformation):
        self._transformations: Tuple[SampleTransformation, ...] = args

    def __call__(self, sample: Sample) -> Sample:
        for transformation in self._transformations:
            sample = transformation(sample)
        return sample


def hr_shape_to_lr_shape(
        hr_shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    if hr_shape[0] % 3 or hr_shape[1] % 3:
        raise ValueError('HR dimensions not divisible by three.')
    return hr_shape[0]//3, hr_shape[1]//3, hr_shape[2]


class ProbaDataGenerator(keras.utils.Sequence):
    def __init__(self,
                 paths: List[SamplePaths],
                 preprocessor: ProbaImagePreprocessor,
                 hr_shape=(378, 378, 1),
                 batch_size=32,
                 include_masks=True,
                 shuffle=True):
        self._paths = paths
        self._preprocessor = preprocessor
        self._batch_size = batch_size
        self._hr_shape = hr_shape
        self._lr_shape = hr_shape_to_lr_shape(hr_shape)
        self._include_masks = include_masks
        self._shuffle = shuffle

        self.on_epoch_end()

    def __len__(self):
        'Number of batches per epoch'
        return int(math.floor(len(self._paths) / self._batch_size))

    def __getitem__(self, index: int):
        paths_in_batch = self._paths[
            index * self._batch_size:(index + 1) * self._batch_size]
        return self._generate_batch(paths_in_batch)

    def to_array(self) -> np.ndarray:
        ret = []
        # Keras internals force usage of range(len()) here
        for batch_iter in range(len(self)):
            ret.append(self[batch_iter])
        return np.concatenate(ret)

    def to_lr_array(self):
        ret = []
        # Keras internals force usage of range(len()) here
        for batch_iter in range(len(self)):
            ret.append(self[batch_iter][SampleEl.LR])
        return np.concatenate(ret)

    def on_epoch_end(self):
        if self._shuffle is True:
            random.shuffle(self._paths)

    def _generate_batch(self, paths: List[SamplePaths]):
        x, y, y_masks = self._init_batch()
        for i, path in enumerate(paths):
            sample = self._preprocessor(self._load_sample(path))
            x[i], y[i], y_masks[i] = sample

        return (x, y, y_masks) if self._include_masks else (x, y)

    def _init_batch(self):
        x = np.empty((self._batch_size, *self._hr_shape))
        y = np.empty((self._batch_size, *self._lr_shape))
        y_masks = np.empty((self._batch_size, *self._lr_shape))
        return x, y, y_masks

    @staticmethod
    def _load_sample(sample_paths: SamplePaths):
        x = load_proba_img_as_array(sample_paths[SampleEl.HR])
        y = load_proba_img_as_array(sample_paths[SampleEl.LR])
        y_masks = load_proba_mask_as_array(sample_paths[SampleEl.LR_MASK])
        return x, y, y_masks


def load_proba_img_as_array(path: Path) -> np.ndarray:
    img = np.expand_dims(io.imread(path, as_gray=False), axis=2)
    img = np.clip(img, a_min=None, a_max=pow(2, 14))
    img = exposure.rescale_intensity(
        img, in_range='uint14', out_range=(0.0, 1.0))
    return img


def load_proba_mask_as_array(path: Path) -> np.ndarray:
    img = np.expand_dims(io.imread(path), axis=2)
    return img_as_float(img)


def show_demo_sample():
    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v_registered'),
        dataset=Dataset.NIR,
        subset=Subset.TRAIN,
        splits={'train': 0.7, 'val': 0.3},
        limit_per_scene=3)
    preprocessor = ProbaImagePreprocessor(
        ProbaHistMatcher())
    train_gen = ProbaDataGenerator(
        dir_scanner.get_split('train'), preprocessor)

    print(dir_scanner)

    batch, sample_in_batch = 0, 0
    plt.figure()
    plt.title('HR')
    plt.imshow(train_gen[batch][SampleEl.HR][sample_in_batch])

    plt.figure()
    plt.title('LR')
    plt.imshow(train_gen[batch][SampleEl.LR][sample_in_batch])

    plt.figure()
    plt.title('LR_MASK')
    plt.imshow(train_gen[batch][SampleEl.LR_MASK][sample_in_batch])

    plt.show()


if __name__ == '__main__':
    show_demo_sample()
