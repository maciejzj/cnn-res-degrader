#!/usr/bin/env python3


"""
This script creates registered proba from the original one using cross
corerlation.
"""

from skimage.registration import phase_cross_correlation
from skimage.transform import resize
from skimage import io
from scipy.ndimage import shift
import numpy as np

from pathlib import Path

def crop_border(img: np.ndarray, margin: float) -> np.ndarray:
    return img[margin:-margin, margin:-margin]

progress_iterator = 0

for hr_path in Path('ProbaV/train').glob('*/**/*hr.png*'):
    hr = io.imread(hr_path)
    scene_dir = hr_path.parent
    export_scene_dir = Path(str(hr_path).replace('ProbaV', 'proba-v_shifted')).parent
    qm_lr_dir = scene_dir / 'masks' / 'lr'
    for lr_path in scene_dir.glob('lr/LR*'):
        lr = io.imread(lr_path)
        
        lr2hr = resize(lr, hr.shape)
        s = phase_cross_correlation(lr2hr, hr, upsample_factor=100)[0]
        shifted_hr = shift(hr, s)
        
        qm_lr_path = qm_lr_dir / str(lr_path.name).replace('LR', 'QM')
        qm_lr = io.imread(qm_lr_path)
        
        new_hr_path = export_scene_dir / str(lr_path.name).replace('LR', 'HR')
        new_lr_path =  export_scene_dir / lr_path.name
        new_qm_lr_path =  export_scene_dir / str(lr_path.name).replace('LR', 'QR')
        
        new_hr_path.parent.mkdir(parents=True, exist_ok=True)
        new_lr_path.parent.mkdir(parents=True, exist_ok=True)
        new_qm_lr_path.parent.mkdir(parents=True, exist_ok=True)
        
        io.imsave(new_hr_path, crop_border(shifted_hr, 3))
        io.imsave(new_lr_path, crop_border(lr, 1))
        io.imsave(new_qm_lr_path, crop_border(qm_lr, 1))

        print(progress_iterator := progress_iterator + 1)
