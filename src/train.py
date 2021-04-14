import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard)

from data_loading import (
    Dataset,
    Subset,
    InterpolationMode,
    ProbaDataGenerator,
    ProbaDirectoryScanner,
    ProbaImagePreprocessor,
    ProbaHistEqualizer,
    ProbaResizer)
from metrics import make_psnr_metric, make_ssim_metric
from models import SimpleConv


TRAIN_LOSSES = {
    'MAE': keras.losses.MeanAbsoluteError(),
    'MSE': keras.losses.MeanSquaredError(),
    'PSNR': make_psnr_metric(),
    'SSIM': make_ssim_metric()}


class Training:
    def __init__(self, model: keras.Model, loss: Callable):
        self._model = model
        self._loss = loss
        self._train_tim = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
        self._callbacks: List[tf.keras.callbacks.Callback] = []

    def make_callbacks(self, callbacks_params: Dict[str, Any]):
        if callbacks_params['tensorboard']:
            self._callbacks.append(TensorBoard(
                log_dir='log/fit-' + self._train_tim))

        if callbacks_params['earlystopping']:
            self._callbacks.append(EarlyStopping(
                monitor='val_loss',
                mode='min',
                min_delta=0.001,
                patience=5,
                verbose=1))

        if callbacks_params['modelcheckpoint']:
            self._callbacks.append(ModelCheckpoint(
                'log/model-' + self._train_tim + '.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1))

    def train(self,
              train_ds: keras.utils.Sequence,
              val_ds: keras.utils.Sequence,
              epochs: int):
        self._model.compile(loss=self._loss,
                            optimizer=keras.optimizers.Adam(),
                            metrics=[
                                keras.metrics.MeanAbsoluteError(),
                                keras.metrics.MeanSquaredError()]
                            )
        self._model.fit(train_ds,
                        validation_data=val_ds,
                        epochs=epochs,
                        callbacks=self._callbacks)
        return self._model


def make_preprocessor(prep_params: Dict[str, Any]) -> ProbaImagePreprocessor:
    transformations = []

    if prep_params['equalize_hist']:
        transformations.append(ProbaHistEqualizer())

    if prep_params['artificial_lr']:
        transformations.append(ProbaResizer(
            prep_params['interpolation_mode']))

    return ProbaImagePreprocessor(*transformations)


def make_params(params_path: Path) -> Dict[str, Any]:
    with open(params_path) as params_file:
        params = yaml.load(params_file, Loader=yaml.FullLoader)

    params['load']['dataset'] = Dataset[params['load']['dataset']]
    prep = params['load']['preprocess']
    prep['interpolation_mode'] = InterpolationMode[prep['interpolation_mode']]
    params['train']['loss'] = TRAIN_LOSSES[params['train']['loss']]

    return params


if __name__ == '__main__':
    params = make_params(Path('params.yaml'))

    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v11_shifted'),
        dataset=params['load']['dataset'],
        subset=Subset.TRAIN,
        splits={'train': params['load']['validation_split'],
                'val': 1.0 - params['load']['validation_split']},
        limit_per_scene=params['load']['limit_per_scene'])
    preprocessor = make_preprocessor(params['load']['preprocess'])

    train_ds = ProbaDataGenerator(
        dir_scanner.get_split('train'),
        preprocessor=preprocessor)

    val_ds = ProbaDataGenerator(
        dir_scanner.get_split('val'),
        preprocessor=preprocessor)

    model = SimpleConv(params['train']['use_lr_masks'])

    training = Training(model, params['train']['loss'])
    training.train(train_ds, val_ds, params['train']['epochs'])

# if __name__ == '__main__':
#     config = yaml.load('params.yaml')

#     dir_scanner = ProbaDirectoryScanner(
#         Path('data/proba-v11_shifted'),
#         dataset=Dataset.NIR,
#         subset=Subset.TRAIN,
#         splits={'train': 0.7, 'val': 0.3},
#         limit_per_scene=3)
#     preprocessor = ProbaImagePreprocessor(
#         ProbaHistEqualizer())

#     train_ds = ProbaDataGenerator(
#         dir_scanner.get_split('train'),
#         preprocessor=preprocessor)

#     val_ds = ProbaDataGenerator(
#         dir_scanner.get_split('val'),
#         preprocessor=preprocessor)

#     model = SimpleConv(use_lr_masks=False)
#     model.compile(loss=make_ssim_metric(), run_eagerly=True)
#     model.fit(train_ds, validation_data=val_ds)
