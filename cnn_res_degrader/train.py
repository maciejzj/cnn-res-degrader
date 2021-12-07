import yaml
import argparse
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple, Union

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard)
from tensorflow.python.ops.numpy_ops import np_config

from cnn_res_degrader.data_loading import (
    Dataset,
    Subset,
    InterpolationMode,
    ProbaDataGenerator,
    ProbaDirectoryScanner,
    ProbaImagePreprocessor,
    ProbaHistMatcher,
    ProbaHrToLrResizer)
from cnn_res_degrader.metrics import make_ssim_metric
from cnn_res_degrader.models import make_model, Models, Gan
from cnn_res_degrader.utils import enable_gpu_if_possible
from cnn_res_degrader.callbacks import InferenceImagePreview


np_config.enable_numpy_behavior()


TRAIN_LOSSES = {
    'BINARY_CROSSENTROPY': keras.losses.BinaryCrossentropy(),
    'MAE': keras.losses.MeanAbsoluteError(),
    'MSE': keras.losses.MeanSquaredError(),
    'SSIM': make_ssim_metric()}


class Training:
    def __init__(self,
                 model: keras.Model,
                 lr: Union[float, Dict[str, float]],
                 loss: Callable):
        self._model = model
        self._lr = lr
        self._loss = loss
        self._train_tim = datetime.now().strftime("%y-%m-%d-%H:%M:%S")
        self._callbacks: List[tf.keras.callbacks.Callback] = []

    def make_callbacks(self, callbacks_params: Dict[str, Any]):
        if callbacks_params['tensorboard']:
            log_dir = Path(f'data/models/{self._model.name}')
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
            log_dir = Path(f'data/models/{self._model.name}')
            name = f'model-{self._model.name}-{self._train_tim}_' + 'e{epoch}.h5'
            log_dir.mkdir(parents=True, exist_ok=True)
            self._callbacks.append(ModelCheckpoint(
                log_dir/name,
                monitor='val_loss',
                mode='min',
                save_best_only=callbacks_params['save_best_only'],
                save_weights_only=True,
                verbose=1))

        if callbacks_params['preview']:
            log_dir = Path(f'data/models/{self._model.name}/'
                           f'preview-{self._model.name}-{self._train_tim}')
            log_dir.mkdir(parents=True, exist_ok=True)
            self._callbacks.append(InferenceImagePreview(
                hr_path=Path(
                    'data/proba-v_registered/test/NIR/imgset0596/HR000.png'),
                lr_path=Path(
                    'data/proba-v_registered/test/NIR/imgset0596/LR000.png'),
                output_dir=log_dir))

    def train(self,
              train_ds: keras.utils.Sequence,
              val_ds: keras.utils.Sequence,
              epochs: int) -> keras.Model:
        if type(self._model) is Gan:
            d_lr = self._lr['discriminator']
            g_lr = self._lr['generator']
            self._model.compile(
                loss_fn=self._loss,
                d_optimizer=keras.optimizers.Adam(learning_rate=g_lr),
                g_optimizer=keras.optimizers.Adam(learning_rate=d_lr))
        else:
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

    if prep_params['match_hist']:
        transformations.append(ProbaHistMatcher())

    if prep_params['artificial_lr']:
        transformations.append(ProbaHrToLrResizer(
            prep_params['interpolation_mode']))

    return ProbaImagePreprocessor(*transformations)


def make_training_data(
        dataset: Dataset,
        input_shape: Tuple[int, int, int],
        batch_size: int,
        validation_split: float,
        preprocessor_params: Dict[str, Any],
        limit_per_scene: int) -> Tuple[ProbaDataGenerator, ProbaDataGenerator]:

    dir_scanner = ProbaDirectoryScanner(
        Path('data/proba-v_registered_a'),
        dataset=dataset,
        subset=Subset.TRAIN,
        splits={'train': validation_split, 'val': 1.0 - validation_split},
        limit_per_scene=limit_per_scene)
    preprocessor = make_preprocessor(preprocessor_params)

    train_ds = ProbaDataGenerator(
        dir_scanner.get_split('train'),
        hr_shape=input_shape,
        batch_size=batch_size,
        preprocessor=preprocessor)

    val_ds = ProbaDataGenerator(
        dir_scanner.get_split('val'),
        hr_shape=input_shape,
        batch_size=batch_size,
        preprocessor=preprocessor)

    return train_ds, val_ds


def make_params(params_path: Path, model_type: Models) -> Dict[str, Any]:
    with open(params_path) as params_file:
        params = yaml.load(params_file, Loader=yaml.FullLoader)[
            model_type.name]

    params['load']['dataset'] = Dataset[params['load']['dataset']]
    prep = params['load']['preprocess']
    prep['interpolation_mode'] = InterpolationMode[prep['interpolation_mode']]
    params['train']['loss'] = TRAIN_LOSSES[params['train']['loss']]

    return params


def train(model_type: Models, training_name: str):
    params = make_params(Path('params.yaml'), model_type)

    model = make_model(
        model_type,
        params['load']['input_shape'],
        name=f'{model_type.name}-{training_name}',
        use_lr_masks=params['train']['use_lr_masks'])

    train_ds, val_ds = make_training_data(
        params['load']['dataset'],
        params['load']['input_shape'],
        params['load']['batch_size'],
        params['load']['validation_split'],
        params['load']['preprocess'],
        params['load']['limit_per_scene'])

    if type(model) is not Gan:
        model.get_functional().summary()

    training = Training(
        model,
        params['train']['lr'],
        params['train']['loss'])
    training.make_callbacks(params['train']['callbacks'])

    training.train(train_ds, val_ds, params['train']['epochs'])


def main():
    enable_gpu_if_possible()

    parser = argparse.ArgumentParser(description='Train networks.')

    parser.add_argument('-s', '--simple', action='store_true',
                        help='Train simple conv net.')
    parser.add_argument('-a', '--autoencoder', action='store_true',
                        help='Train autoencoder et.')
    parser.add_argument('-g', '--gan', action='store_true',
                        help='Train gan net.')
    parser.add_argument('training_name')

    args = parser.parse_args()

    if args.simple:
        train(Models.SIMPLE_CONV, args.training_name)
    if args.autoencoder:
        train(Models.AUTOENCODER, args.training_name)
    if args.gan:
        train(Models.GAN, args.training_name)


if __name__ == '__main__':
    main()
