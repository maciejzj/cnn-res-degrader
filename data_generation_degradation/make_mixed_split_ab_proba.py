from pathlib import Path
from sklearn.model_selection import train_test_split
import shutil


def split_proba_v_train_set(proba_v_dir_path: Path):
    dataset_type_dir_names = ('NIR', 'RED')
    for dataset_type_dir_name in dataset_type_dir_names:
        train_dir = proba_v_dir_path/'train'/dataset_type_dir_name
        train_scene_dirs = list(train_dir.glob('imgset*'))

        '''
        Don't be confused by the naming; the sklern function is named
        'train_test_split' but we use it to split the training set into two.
        '''
        splits = train_test_split(
            train_scene_dirs, train_size=0.5, random_state=42)

        split_dataset_suffixes = ('_a', '_b')

        for split, suffix in zip(splits, split_dataset_suffixes):
            split_proba_v_dir_path = proba_v_dir_path.with_name(
                proba_v_dir_path.name + suffix)

            for scene_dir in split:
                dest_dir = split_proba_v_dir_path.joinpath(
                    *scene_dir.parts[2:])
                shutil.copytree(scene_dir, dest_dir)

            test_dir = proba_v_dir_path/'test'/dataset_type_dir_name
            dest_dir = split_proba_v_dir_path/'test'/dataset_type_dir_name
            shutil.copytree(test_dir, dest_dir)


def main():
    split_proba_v_train_set(Path('data/proba-v_registered'))


if __name__ == '__main__':
    main()
