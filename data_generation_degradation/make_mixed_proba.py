import re
import random
import shutil
from pathlib import Path

"""
This is a script for mixing randomly images from different datasets into one.
"""


def subsitute_dataset_for_lr_path(lr_path, new_ds_name):
    if not re.match(r'LR[0-9]*\.png', lr_path.name):
        raise ValueError('Path for dataset substitution should lead to'
                         'LR image, but it does not.')

    new_path = list(lr_path.parts)
    new_path[-7] = new_ds_name
    return Path(*new_path)


if __name__ == '__main__':
    random.seed(42)
    dataset_names = [
        'proba-v_model-AUTOENCODER-dvc-21-09-21-145004_e34_b',
        'proba-v_model-AUTOENCODERUNET-dvc-21-10-20-074000_e20_b',
        'proba-v_model-GAN-dvc-21-09-21-134739_e68_b',
        'proba-v_model-SIMPLE_CONV-adapthist-dvc-21-10-14-120809_e13_b',
        'proba-v_model-SIMPLE_CONV-matchhist-dvc-21-10-14-151212_e40_b'
    ]
    datasets_root_dir = Path(
        '/Volumes/pub/Teams/Projekty ML/SRR/datasets/3 DeepSent/'
        'ProbaV_artificial/')
    output_ds = 'proba-v_mixed_b'

    dataset_paths = [datasets_root_dir/ds_name for ds_name in dataset_names]

    reference_ds_dir = dataset_paths[0]
    for scene_dir in reference_ds_dir.glob('*/**/imgset*'):
        lr_dir = scene_dir/'lr'
        output_scene_path = None
        for lr_img in lr_dir.glob('LR*.png'):
            source_ds = random.choice(dataset_names)
            source_lr_path = subsitute_dataset_for_lr_path(lr_img, source_ds)
            output_lr_path = subsitute_dataset_for_lr_path(lr_img, output_ds)
            output_lr_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(source_lr_path, output_lr_path)
            output_scene_path = output_lr_path.parent.parent
            print('-------------')
            print(source_lr_path)
            print(output_lr_path)
            print(output_scene_path)
        shutil.copy(scene_dir/'hr.png', output_scene_path)


def test_substitutes_dataset_correctly():
    lr_path = Path('/Volumes/pub/Teams/Projekty ML/SRR/datasets/3 DeepSent/'
                   'ProbaV_artificial/proba-v_bicubic_b/ProbaV/train/NIR/'
                   'imgset0648/lr/LR000.png')
    new_ds_name = 'NEW_DS'
    expected = Path('/Volumes/pub/Teams/Projekty ML/SRR/datasets/3 DeepSent/'
                    'ProbaV_artificial/NEW_DS/ProbaV/train/NIR/'
                    'imgset0648/lr/LR000.png')

    assert subsitute_dataset_for_lr_path(lr_path, new_ds_name) == expected
    