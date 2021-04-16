import sys
import yaml
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard)

from cnn_res_degrader.data_loading import (
    Dataset,
    Subset,
    InterpolationMode,
    ProbaDataGenerator,
    ProbaDirectoryScanner,
    ProbaImagePreprocessor,
    ProbaHistEqualizer,
    ProbaHrToLrResizer)
from cnn_res_degrader.metrics import make_ssim_metric
from cnn_res_degrader.models import SimpleConv


TRAIN_LOSSES = {
    'MAE': keras.losses.MeanAbsoluteError(),
    'MSE': keras.losses.MeanSquaredError(),
    'SSIM': make_ssim_metric()}


class Training:
    def __init__(self, model: keras.Model, lr: float, loss: Callable):
        self._model = model
        self._lr = lr
        self._loss = loss
        self._train_tim = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
        self._callbacks: List[tf.keras.callbacks.Callback] = []

    def make_callbacks(self, callbacks_params: Dict[str, Any]):
        if callbacks_params['tensorboard']:
            self._callbacks.append(TensorBoard(
                log_dir=f'log/fit-{self._model.name}-{self._train_tim}'))

        if callbacks_params['earlystopping']:
            self._callbacks.append(EarlyStopping(
                monitor='val_loss',
                mode='min',
                min_delta=callbacks_params['stopping_delta'],
                patience=callbacks_params['stopping_patience'],
                verbose=1))

        if callbacks_params['modelcheckpoint']:
            self._callbacks.append(ModelCheckpoint(
                f'log/model-{self._model.name}-{self._train_tim}.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1))

    def train(self,
              train_ds: keras.utils.Sequence,
              val_ds: keras.utils.Sequence,
              epochs: int):
        self._model.compile(loss=self._loss,
                            optimizer=keras.optimizers.Adam(
                                learning_rate=self._lr),
                            metrics=[
                                keras.metrics.MeanAbsoluteError(),
                                keras.metrics.MeanSquaredError()])
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
        transformations.append(ProbaHrToLrResizer(
            prep_params['interpolation_mode']))

    return ProbaImagePreprocessor(*transformations)


def make_training_data(load_params: Dict[str, Any]) -> Tuple[
        ProbaDataGenerator, ProbaDataGenerator]:
    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v11_shifted'),
        dataset=load_params['dataset'],
        subset=Subset.TRAIN,
        splits={'train': load_params['validation_split'],
                'val': 1.0 - load_params['validation_split']},
        limit_per_scene=load_params['limit_per_scene'])
    preprocessor = make_preprocessor(load_params['preprocess'])

    train_ds = ProbaDataGenerator(
        dir_scanner.get_split('train'),
        hr_shape=load_params['input_shape'],
        preprocessor=preprocessor)

    val_ds = ProbaDataGenerator(
        dir_scanner.get_split('val'),
        hr_shape=load_params['input_shape'],
        preprocessor=preprocessor)

    return train_ds, val_ds


def make_params(params_path: Path) -> Dict[str, Any]:
    with open(params_path) as params_file:
        params = yaml.load(params_file, Loader=yaml.FullLoader)

    params['load']['dataset'] = Dataset[params['load']['dataset']]
    prep = params['load']['preprocess']
    prep['interpolation_mode'] = InterpolationMode[prep['interpolation_mode']]
    params['train']['loss'] = TRAIN_LOSSES[params['train']['loss']]

    return params


def main():
    params = make_params(Path('params.yaml'))
    train_ds, val_ds = make_training_data(params['load'])
    model = SimpleConv(params['load']['input_shape'],
                       name=f'{params["model"]}-{sys.argv[1]}',
                       use_lr_masks=params['train']['use_lr_masks'])
    model.get_functional().summary()

    training = Training(model, params['train']['lr'], params['train']['loss'])
    training.make_callbacks(params['train']['callbacks'])
    training.train(train_ds, val_ds, params['train']['epochs'])


if __name__ == '__main__':
    main()
