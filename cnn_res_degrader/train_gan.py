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
from cnn_res_degrader.models import make_model, Models, Gan


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
            log_dir = Path('log')
            log_dir.mkdir(parents=True, exist_ok=True)
            self._callbacks.append(TensorBoard(
                log_dir=log_dir/f'fit-{self._model.name}-{self._train_tim}'))

        if callbacks_params['earlystopping']:
            self._callbacks.append(EarlyStopping(
                monitor='val_loss',
                mode='min',
                min_delta=callbacks_params['stopping_delta'],
                patience=callbacks_params['stopping_patience'],
                verbose=1))

        if callbacks_params['modelcheckpoint']:
            log_dir = Path('data/models')
            log_dir.mkdir(parents=True, exist_ok=True)
            self._callbacks.append(ModelCheckpoint(
                log_dir/f'model-{self._model.name}-{self._train_tim}.h5',
                monitor='val_loss',
                mode='min',
                save_best_only=True,
                verbose=1))

    def train(self,
              train_ds: keras.utils.Sequence,
              val_ds: keras.utils.Sequence,
              epochs: int):
        self._model.compile()
        self._model._discriminator.compile(loss=self._loss,
                                           optimizer=keras.optimizers.Adam(
                                               learning_rate=self._lr),
                                           metrics=[
                                               keras.metrics.MeanAbsoluteError(),
                                               keras.metrics.MeanSquaredError()])
        self._model._generator.compile(loss=self._loss,
                                       optimizer=keras.optimizers.Adam(
                                           learning_rate=self._lr),
                                       metrics=[
                                           keras.metrics.MeanAbsoluteError(),
                                           keras.metrics.MeanSquaredError()])
        self._model.fit(train_ds, epochs=epochs)
        return self._model


def make_preprocessor(prep_params: Dict[str, Any]) -> ProbaImagePreprocessor:
    transformations = []

    if prep_params['equalize_hist']:
        transformations.append(ProbaHistEqualizer())

    if prep_params['artificial_lr']:
        transformations.append(ProbaHrToLrResizer(
            prep_params['interpolation_mode']))

    return ProbaImagePreprocessor(*transformations)


def make_training_data(
        dataset: Dataset,
        input_shape: Tuple[int, int, int],
        validation_split: float,
        preprocessor_params: Dict[str, Any],
        limit_per_scene: int) -> Tuple[ProbaDataGenerator, ProbaDataGenerator]:

    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v11_shifted'),
        dataset=dataset,
        subset=Subset.TRAIN,
        splits={'train': validation_split, 'val': 1.0 - validation_split},
        limit_per_scene=limit_per_scene)
    preprocessor = make_preprocessor(preprocessor_params)

    train_ds = ProbaDataGenerator(
        dir_scanner.get_split('train'),
        hr_shape=input_shape,
        preprocessor=preprocessor)

    val_ds = ProbaDataGenerator(
        dir_scanner.get_split('val'),
        hr_shape=input_shape,
        preprocessor=preprocessor)

    return train_ds, val_ds


def make_params(params_path: Path) -> Dict[str, Any]:
    with open(params_path) as params_file:
        params = yaml.load(params_file, Loader=yaml.FullLoader)

    params['model'] = Models[params['model']]
    params['load']['dataset'] = Dataset[params['load']['dataset']]
    prep = params['load']['preprocess']
    prep['interpolation_mode'] = InterpolationMode[prep['interpolation_mode']]
    params['train']['loss'] = TRAIN_LOSSES[params['train']['loss']]

    return params


def main():
    params = make_params(Path('params.yaml'))

    train_ds, val_ds = make_training_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['validation_split'],
        params['load']['preprocess'],
        params['load']['limit_per_scene'])

    model = Gan.make_with_simple_conv_gen(params['load']['input_shape'])
    training = Training(
        model,
        params['train']['lr'],
        params['train']['loss'])
    training.make_callbacks(params['train']['callbacks'])

    training.train(train_ds, val_ds, params['train']['epochs'])


if __name__ == '__main__':
    main()