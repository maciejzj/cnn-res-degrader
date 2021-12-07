import sys
from pathlib import Path

from skimage import io, exposure
from matplotlib import pyplot as plt


def save_vis(scene_path, output_path):
    hr_path = scene_path/'hr.png'
    lr_paths = list((scene_path/'lr').glob('*.png'))
    paths = [hr_path] + lr_paths

    plt.figure()
    for i, path in enumerate(paths):
        plt.subplot(2, 5, i + 1)
        img = io.imread(path)
        img = exposure.rescale_intensity(
            img, in_range='uint14', out_range=(0.0, 1.0))
        plt.imshow(img)
        plt.title(path.name)

    plt.show()


def main():
    save_vis(Path(sys.argv[1]), Path(sys.argv[2]))


if __name__ == '__main__':
    main()
